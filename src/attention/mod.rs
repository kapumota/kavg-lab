use crate::config::{
    AgentObjective, AgentSweepConfig, AttentionConfig, AttentionMaskConfig, AttentionSolverConfig,
    AttentionSolverMethod, MaskEntry, MultiHeadAttentionConfig,
};
use crate::math::{dot, format_vector, norm2, sub};
use anyhow::{bail, Result};

/// Resultado de atención para una consulta individual.
#[derive(Debug, Clone)]
pub struct AttentionResult {
    pub index: usize,
    pub solver_method: String,
    pub query: Vec<f64>,
    pub scores: Vec<f64>,
    pub masked_scores: Vec<f64>,
    pub prior: Vec<f64>,
    pub standard_weights: Vec<f64>,
    pub regularized_weights: Vec<f64>,
    pub standard_output: Vec<f64>,
    pub regularized_output: Vec<f64>,
    pub weight_l1_distance: f64,
    pub weight_l2_distance: f64,
    pub output_l2_distance: f64,
    pub standard_entropy: f64,
    pub regularized_entropy: f64,
    pub kl_regularized_to_softmax: f64,
    pub kl_regularized_to_prior: f64,
    pub js_softmax_regularized: f64,
    pub effective_tokens_standard: f64,
    pub effective_tokens_regularized: f64,
    pub standard_top1_mass: f64,
    pub regularized_top1_mass: f64,
    pub standard_topk_mass: f64,
    pub regularized_topk_mass: f64,
    pub iterations: usize,
    pub solver_metric: f64,
}

impl AttentionResult {
    pub fn query_csv(&self) -> String {
        format_vector(&self.query)
    }

    pub fn scores_csv(&self) -> String {
        format_vector_special(&self.scores)
    }

    pub fn masked_scores_csv(&self) -> String {
        format_vector_special(&self.masked_scores)
    }

    pub fn prior_csv(&self) -> String {
        format_vector(&self.prior)
    }

    pub fn standard_weights_csv(&self) -> String {
        format_vector(&self.standard_weights)
    }

    pub fn regularized_weights_csv(&self) -> String {
        format_vector(&self.regularized_weights)
    }

    pub fn standard_output_csv(&self) -> String {
        format_vector(&self.standard_output)
    }

    pub fn regularized_output_csv(&self) -> String {
        format_vector(&self.regularized_output)
    }
}

/// Resultado compacto por cabecera para la demo multi-head.
#[derive(Debug, Clone)]
pub struct HeadAttentionResult {
    pub head_index: usize,
    pub head_name: String,
    pub query_index: usize,
    pub temperature: f64,
    pub kernel_gamma: f64,
    pub prior: Vec<f64>,
    pub standard_weights: Vec<f64>,
    pub regularized_weights: Vec<f64>,
    pub regularized_output: Vec<f64>,
    pub entropy: f64,
    pub effective_tokens: f64,
    pub top1_mass: f64,
    pub topk_mass: f64,
    pub iterations: usize,
    pub solver_metric: f64,
}

/// Resultado agregado por query para multi-head attention.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionResult {
    pub query_index: usize,
    pub query: Vec<f64>,
    pub aggregated_output: Vec<f64>,
    pub average_entropy: f64,
    pub mean_pairwise_l1: f64,
    pub mean_pairwise_l2: f64,
    pub mean_pairwise_js: f64,
    pub heads: Vec<HeadAttentionResult>,
}

impl MultiHeadAttentionResult {
    pub fn query_csv(&self) -> String {
        format_vector(&self.query)
    }

    pub fn aggregated_output_csv(&self) -> String {
        format_vector(&self.aggregated_output)
    }

    pub fn head_names_csv(&self) -> String {
        self.heads
            .iter()
            .map(|h| h.head_name.clone())
            .collect::<Vec<_>>()
            .join("|")
    }

    pub fn head_weights_csv(&self) -> String {
        self.heads
            .iter()
            .map(|h| format!("{}={}", h.head_name, format_vector(&h.regularized_weights)))
            .collect::<Vec<_>>()
            .join("|")
    }

    pub fn head_outputs_csv(&self) -> String {
        self.heads
            .iter()
            .map(|h| format!("{}={}", h.head_name, format_vector(&h.regularized_output)))
            .collect::<Vec<_>>()
            .join("|")
    }

    pub fn head_entropies_csv(&self) -> String {
        self.heads
            .iter()
            .map(|h| format!("{}={:.10}", h.head_name, h.entropy))
            .collect::<Vec<_>>()
            .join("|")
    }
}

/// Resultado de una configuración evaluada por el barrido experimental.
#[derive(Debug, Clone)]
pub struct AgentSweepResult {
    pub rank: usize,
    pub gamma: f64,
    pub temperature: f64,
    pub prior_name: String,
    pub objective: String,
    pub score: f64,
    pub mean_regularized_entropy: f64,
    pub mean_distance_to_prior: f64,
    pub mean_difference_from_softmax: f64,
    pub mean_output_shift: f64,
    pub mean_js_softmax_regularized: f64,
    pub mean_effective_tokens: f64,
}

#[derive(Clone, Copy)]
struct AttentionParams<'a> {
    temperature: f64,
    kernel_gamma: f64,
    solver: &'a AttentionSolverConfig,
    prior: Option<&'a Vec<f64>>,
    mask: Option<&'a AttentionMaskConfig>,
    top_k: usize,
}

/// Ejecuta la demo completa para todas las consultas del archivo YAML.
pub fn run_attention_demo(config: &AttentionConfig) -> Result<Vec<AttentionResult>> {
    config.validate()?;
    let top_k = config.top_k.unwrap_or(config.keys.len().min(3));
    let params = AttentionParams {
        temperature: config.temperature,
        kernel_gamma: config.kernel_gamma,
        solver: &config.attention_solver,
        prior: config.prior.as_ref(),
        mask: config.mask.as_ref(),
        top_k,
    };

    let mut results = Vec::new();
    for (index, query) in config.queries.iter().enumerate() {
        results.push(run_single_attention(
            index,
            query,
            &config.keys,
            &config.values,
            params,
        )?);
    }

    Ok(results)
}

/// Ejecuta una demo de multi-head attention con varias cabeceras configurables.
pub fn run_multihead_attention_demo(
    config: &MultiHeadAttentionConfig,
) -> Result<Vec<MultiHeadAttentionResult>> {
    config.validate()?;
    let top_k = config.top_k.unwrap_or(config.keys.len().min(3));
    let mut results = Vec::new();

    for (query_index, query) in config.queries.iter().enumerate() {
        let mut heads = Vec::new();

        for (head_index, head) in config.heads.iter().enumerate() {
            let solver = head
                .attention_solver
                .as_ref()
                .unwrap_or(&config.default_attention_solver);
            let mask = head.mask.as_ref().or(config.default_mask.as_ref());
            let params = AttentionParams {
                temperature: head.temperature,
                kernel_gamma: head.kernel_gamma,
                solver,
                prior: head.prior.as_ref(),
                mask,
                top_k,
            };
            let single =
                run_single_attention(query_index, query, &config.keys, &config.values, params)?;
            let head_name = head
                .name
                .clone()
                .unwrap_or_else(|| format!("head_{}", head_index));

            heads.push(HeadAttentionResult {
                head_index,
                head_name,
                query_index,
                temperature: head.temperature,
                kernel_gamma: head.kernel_gamma,
                prior: single.prior,
                standard_weights: single.standard_weights,
                regularized_weights: single.regularized_weights,
                regularized_output: single.regularized_output,
                entropy: single.regularized_entropy,
                effective_tokens: single.effective_tokens_regularized,
                top1_mass: single.regularized_top1_mass,
                topk_mass: single.regularized_topk_mass,
                iterations: single.iterations,
                solver_metric: single.solver_metric,
            });
        }

        let aggregated_output = average_outputs(&heads);
        let average_entropy = heads.iter().map(|h| h.entropy).sum::<f64>() / heads.len() as f64;
        let (mean_pairwise_l1, mean_pairwise_l2, mean_pairwise_js) = pairwise_head_metrics(&heads);

        results.push(MultiHeadAttentionResult {
            query_index,
            query: query.clone(),
            aggregated_output,
            average_entropy,
            mean_pairwise_l1,
            mean_pairwise_l2,
            mean_pairwise_js,
            heads,
        });
    }

    Ok(results)
}

/// Ejecuta un barrido experimental que prueba configuraciones de atención regularizada.
pub fn run_agent_sweep(config: &AgentSweepConfig) -> Result<Vec<AgentSweepResult>> {
    config.validate()?;

    let mut candidates = Vec::new();
    let priors: Vec<(String, Option<Vec<f64>>)> = match &config.priors {
        Some(named) => named
            .iter()
            .map(|p| (p.name.clone(), Some(p.values.clone())))
            .collect(),
        None => vec![(
            "base-prior".to_string(),
            config.base_attention.prior.clone(),
        )],
    };

    for gamma in &config.gamma_values {
        for temperature in &config.temperature_values {
            for (prior_name, prior) in &priors {
                let mut candidate_config = config.base_attention.clone();
                candidate_config.kernel_gamma = *gamma;
                candidate_config.temperature = *temperature;
                candidate_config.prior = prior.clone();

                let results = run_attention_demo(&candidate_config)?;
                let mean_regularized_entropy = mean(results.iter().map(|r| r.regularized_entropy));
                let mean_distance_to_prior = mean(
                    results
                        .iter()
                        .map(|r| l1_distance(&r.regularized_weights, &r.prior)),
                );
                let mean_difference_from_softmax =
                    mean(results.iter().map(|r| r.weight_l1_distance));
                let mean_output_shift = mean(results.iter().map(|r| r.output_l2_distance));
                let mean_js_softmax_regularized =
                    mean(results.iter().map(|r| r.js_softmax_regularized));
                let mean_effective_tokens =
                    mean(results.iter().map(|r| r.effective_tokens_regularized));
                let score = score_candidate(
                    &config.objective,
                    mean_regularized_entropy,
                    mean_distance_to_prior,
                    mean_difference_from_softmax,
                    mean_output_shift,
                    mean_js_softmax_regularized,
                );

                candidates.push(AgentSweepResult {
                    rank: 0,
                    gamma: *gamma,
                    temperature: *temperature,
                    prior_name: prior_name.clone(),
                    objective: config.objective.as_str().to_string(),
                    score,
                    mean_regularized_entropy,
                    mean_distance_to_prior,
                    mean_difference_from_softmax,
                    mean_output_shift,
                    mean_js_softmax_regularized,
                    mean_effective_tokens,
                });
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let limit = config
        .output_limit
        .unwrap_or(candidates.len())
        .min(candidates.len());
    candidates.truncate(limit);
    for (rank, candidate) in candidates.iter_mut().enumerate() {
        candidate.rank = rank + 1;
    }

    Ok(candidates)
}

fn run_single_attention(
    index: usize,
    query: &[f64],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
    params: AttentionParams<'_>,
) -> Result<AttentionResult> {
    let scores = attention_scores(query, keys);
    let masked_scores = apply_mask_to_scores(index, &scores, params.mask)?;
    let allowed: Vec<bool> = masked_scores.iter().map(|s| s.is_finite()).collect();
    anyhow::ensure!(
        allowed.iter().any(|v| *v),
        "La máscara bloqueó todos los tokens para query #{}.",
        index
    );

    let prior = build_prior(params.prior, keys.len(), &allowed)?;
    let scaled_scores: Vec<f64> = masked_scores
        .iter()
        .map(|s| {
            if s.is_finite() {
                s / params.temperature
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    let standard_weights = softmax(&scaled_scores);
    let standard_output = weighted_sum(&standard_weights, values);

    let regularized = solve_kernel_regularized_attention(
        &masked_scores,
        &prior,
        params.temperature,
        params.kernel_gamma,
        params.solver,
        &allowed,
    );

    let regularized_output = weighted_sum(&regularized.weights, values);
    let diff_weights = sub(&standard_weights, &regularized.weights);
    let diff_outputs = sub(&standard_output, &regularized_output);
    let standard_entropy = entropy(&standard_weights);
    let regularized_entropy = regularized.entropy;
    let standard_top1_mass = top_k_mass(&standard_weights, 1);
    let regularized_top1_mass = top_k_mass(&regularized.weights, 1);
    let standard_topk_mass = top_k_mass(&standard_weights, params.top_k);
    let regularized_topk_mass = top_k_mass(&regularized.weights, params.top_k);

    Ok(AttentionResult {
        index,
        solver_method: params.solver.method().as_str().to_string(),
        query: query.to_vec(),
        scores,
        masked_scores,
        prior,
        standard_weights: standard_weights.clone(),
        regularized_weights: regularized.weights.clone(),
        standard_output,
        regularized_output,
        weight_l1_distance: diff_weights.iter().map(|v| v.abs()).sum(),
        weight_l2_distance: norm2(&diff_weights),
        output_l2_distance: norm2(&diff_outputs),
        standard_entropy,
        regularized_entropy,
        kl_regularized_to_softmax: kl_divergence(&regularized.weights, &standard_weights),
        kl_regularized_to_prior: kl_divergence(&regularized.weights, &regularized.prior),
        js_softmax_regularized: jensen_shannon_distance(&standard_weights, &regularized.weights),
        effective_tokens_standard: standard_entropy.exp(),
        effective_tokens_regularized: regularized_entropy.exp(),
        standard_top1_mass,
        regularized_top1_mass,
        standard_topk_mass,
        regularized_topk_mass,
        iterations: regularized.iterations,
        solver_metric: regularized.solver_metric,
    })
}

/// Calcula scores de atención escalados por dimensión: q · k / sqrt(d).
fn attention_scores(query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
    let scale = (query.len() as f64).sqrt();
    keys.iter().map(|key| dot(query, key) / scale).collect()
}

/// Aplica máscaras tipo Transformer. Una máscara aditiva usa 0 para permitir y -inf para bloquear.
fn apply_mask_to_scores(
    query_index: usize,
    scores: &[f64],
    mask: Option<&AttentionMaskConfig>,
) -> Result<Vec<f64>> {
    let Some(mask) = mask else {
        return Ok(scores.to_vec());
    };

    let mut masked = scores.to_vec();
    match mask {
        AttentionMaskConfig::None => {}
        AttentionMaskConfig::Causal => {
            for (key_index, score) in masked.iter_mut().enumerate() {
                if key_index > query_index {
                    *score = f64::NEG_INFINITY;
                }
            }
        }
        AttentionMaskConfig::Custom { matrix } => {
            let row = &matrix[query_index];
            for (score, entry) in masked.iter_mut().zip(row) {
                let value = entry_to_f64(entry)?;
                if value.is_infinite() && value.is_sign_positive() {
                    bail!(
                        "La máscara no debe usar +inf; use 0 para permitir o -inf para bloquear."
                    );
                }
                if value.is_infinite() && value.is_sign_negative() {
                    *score = f64::NEG_INFINITY;
                } else {
                    *score += value;
                }
            }
        }
    }

    Ok(masked)
}

fn entry_to_f64(entry: &MaskEntry) -> Result<f64> {
    entry.to_f64()
}

/// Softmax estable numéricamente. Los scores -inf reciben peso cero.
pub fn softmax(scores: &[f64]) -> Vec<f64> {
    let finite_values: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();
    if finite_values.is_empty() {
        let n = scores.len();
        return vec![1.0 / n as f64; n];
    }

    let max_score = finite_values.into_iter().fold(f64::NEG_INFINITY, f64::max);
    let exp_values: Vec<f64> = scores
        .iter()
        .map(|s| {
            if s.is_finite() {
                (s - max_score).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum_exp: f64 = exp_values.iter().sum();

    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        let n = scores.len();
        return vec![1.0 / n as f64; n];
    }

    exp_values.iter().map(|v| v / sum_exp).collect()
}

/// Calcula una combinación convexa de los vectores de valor.
fn weighted_sum(weights: &[f64], values: &[Vec<f64>]) -> Vec<f64> {
    let out_dim = values[0].len();
    let mut output = vec![0.0; out_dim];

    for (weight, value) in weights.iter().zip(values) {
        for j in 0..out_dim {
            output[j] += weight * value[j];
        }
    }

    output
}

/// Entropía de Shannon, usando la convención 0 log 0 = 0.
fn entropy(probabilities: &[f64]) -> f64 {
    let eps = 1.0e-15;
    -probabilities
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| p * p.max(eps).ln())
        .sum::<f64>()
}

#[derive(Debug, Clone)]
struct RegularizedAttentionSolution {
    weights: Vec<f64>,
    prior: Vec<f64>,
    entropy: f64,
    iterations: usize,
    solver_metric: f64,
}

/// Resuelve la atención regularizada por kernel sobre el simplex.
///
/// El problema resuelto es:
///
/// min_p  - <scores, p>
///      + temperature * sum_i p_i log(p_i)
///      + gamma / 2 * ||p - prior||²
/// s.t.   p_i >= 0, sum_i p_i = 1,
///        p_i = 0 en posiciones bloqueadas por máscara.
///
/// Se usa gradiente proyectado para mantener el MVP simple y transparente.
fn solve_kernel_regularized_attention(
    scores: &[f64],
    prior: &[f64],
    temperature: f64,
    gamma: f64,
    solver: &AttentionSolverConfig,
    allowed: &[bool],
) -> RegularizedAttentionSolution {
    match solver.method() {
        AttentionSolverMethod::ProjectedGradient => {
            solve_attention_projected_gradient(scores, prior, temperature, gamma, solver, allowed)
        }
        AttentionSolverMethod::MirrorDescent => {
            solve_attention_mirror_descent(scores, prior, temperature, gamma, solver, allowed)
        }
        AttentionSolverMethod::FrankWolfe => {
            solve_attention_frank_wolfe(scores, prior, temperature, gamma, solver, allowed)
        }
    }
}

fn initial_attention_weights(scores: &[f64], temperature: f64, allowed: &[bool]) -> Vec<f64> {
    let scaled_scores: Vec<f64> = scores
        .iter()
        .map(|s| {
            if s.is_finite() {
                s / temperature
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();
    project_to_masked_simplex(&softmax(&scaled_scores), allowed)
}

fn attention_gradient(
    weights: &[f64],
    scores: &[f64],
    prior: &[f64],
    temperature: f64,
    gamma: f64,
    allowed: &[bool],
) -> Vec<f64> {
    let eps = 1.0e-12;
    weights
        .iter()
        .zip(scores)
        .zip(prior)
        .zip(allowed)
        .map(|(((p, s), p0), is_allowed)| {
            if *is_allowed {
                -*s + temperature * ((*p).max(eps).ln() + 1.0) + gamma * (*p - *p0)
            } else {
                0.0
            }
        })
        .collect()
}

fn solve_attention_projected_gradient(
    scores: &[f64],
    prior: &[f64],
    temperature: f64,
    gamma: f64,
    solver: &AttentionSolverConfig,
    allowed: &[bool],
) -> RegularizedAttentionSolution {
    let mut weights = initial_attention_weights(scores, temperature, allowed);
    let mut step = solver.initial_step;
    let min_step = solver.min_step();
    let mut solver_metric = f64::INFINITY;
    let mut iterations = 0;

    for iter in 0..solver.max_iterations {
        iterations = iter + 1;
        let gradient = attention_gradient(&weights, scores, prior, temperature, gamma, allowed);
        let candidate: Vec<f64> = weights
            .iter()
            .zip(&gradient)
            .map(|(p, g)| p - step * g)
            .collect();
        let projected = project_to_masked_simplex(&candidate, allowed);
        let diff = sub(&projected, &weights);
        solver_metric = norm2(&diff);
        weights = projected;

        if solver_metric <= solver.tolerance {
            break;
        }
        step = (step * 0.995).max(min_step);
    }

    RegularizedAttentionSolution {
        entropy: entropy(&weights),
        weights,
        prior: prior.to_vec(),
        iterations,
        solver_metric,
    }
}

fn solve_attention_mirror_descent(
    scores: &[f64],
    prior: &[f64],
    temperature: f64,
    gamma: f64,
    solver: &AttentionSolverConfig,
    allowed: &[bool],
) -> RegularizedAttentionSolution {
    let mut weights = initial_attention_weights(scores, temperature, allowed);
    let mut step = solver.initial_step;
    let min_step = solver.min_step();
    let mut solver_metric = f64::INFINITY;
    let mut iterations = 0;

    for iter in 0..solver.max_iterations {
        iterations = iter + 1;
        let gradient = attention_gradient(&weights, scores, prior, temperature, gamma, allowed);
        let candidate: Vec<f64> = weights
            .iter()
            .zip(&gradient)
            .zip(allowed)
            .map(|((p, g), is_allowed)| {
                if *is_allowed {
                    p.max(1.0e-15) * (-step * g).clamp(-60.0, 60.0).exp()
                } else {
                    0.0
                }
            })
            .collect();
        let projected = project_to_masked_simplex(&candidate, allowed);
        let diff = sub(&projected, &weights);
        solver_metric = norm2(&diff);
        weights = projected;

        if solver_metric <= solver.tolerance {
            break;
        }
        step = (step * 0.995).max(min_step);
    }

    RegularizedAttentionSolution {
        entropy: entropy(&weights),
        weights,
        prior: prior.to_vec(),
        iterations,
        solver_metric,
    }
}

fn solve_attention_frank_wolfe(
    scores: &[f64],
    prior: &[f64],
    temperature: f64,
    gamma: f64,
    solver: &AttentionSolverConfig,
    allowed: &[bool],
) -> RegularizedAttentionSolution {
    let mut weights = initial_attention_weights(scores, temperature, allowed);
    let mut solver_metric = f64::INFINITY;
    let mut iterations = 0;

    for iter in 0..solver.max_iterations {
        iterations = iter + 1;
        let gradient = attention_gradient(&weights, scores, prior, temperature, gamma, allowed);
        let mut best_index = None;
        let mut best_value = f64::INFINITY;
        for (index, (value, is_allowed)) in gradient.iter().zip(allowed).enumerate() {
            if *is_allowed && *value < best_value {
                best_value = *value;
                best_index = Some(index);
            }
        }

        let Some(best_index) = best_index else { break };
        let mut vertex = vec![0.0; weights.len()];
        vertex[best_index] = 1.0;
        let step = (2.0 / (iter as f64 + 2.0)).min(solver.initial_step.max(1.0e-12));
        let next: Vec<f64> = weights
            .iter()
            .zip(&vertex)
            .map(|(p, v)| (1.0 - step) * p + step * v)
            .collect();
        let projected = project_to_masked_simplex(&next, allowed);
        let diff = sub(&projected, &weights);
        solver_metric = norm2(&diff);
        weights = projected;

        if solver_metric <= solver.tolerance {
            break;
        }
    }

    RegularizedAttentionSolution {
        entropy: entropy(&weights),
        weights,
        prior: prior.to_vec(),
        iterations,
        solver_metric,
    }
}

/// Proyección euclidiana al simplex {p >= 0, sum p = 1}.
fn project_to_simplex(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0;
    let mut rho = 0usize;

    for (j, value) in sorted.iter().enumerate() {
        cumulative += value;
        let theta = (cumulative - 1.0) / (j as f64 + 1.0);
        if value - theta > 0.0 {
            rho = j + 1;
        }
    }

    let theta = if rho == 0 {
        0.0
    } else {
        let sum_rho: f64 = sorted.iter().take(rho).sum();
        (sum_rho - 1.0) / rho as f64
    };

    values.iter().map(|v| (v - theta).max(0.0)).collect()
}

/// Proyección al simplex con posiciones prohibidas por máscara.
fn project_to_masked_simplex(values: &[f64], allowed: &[bool]) -> Vec<f64> {
    let allowed_values: Vec<f64> = values
        .iter()
        .zip(allowed)
        .filter_map(|(value, is_allowed)| if *is_allowed { Some(*value) } else { None })
        .collect();
    let projected_allowed = project_to_simplex(&allowed_values);
    let mut projected = vec![0.0; values.len()];
    let mut cursor = 0;
    for (index, is_allowed) in allowed.iter().enumerate() {
        if *is_allowed {
            projected[index] = projected_allowed[cursor];
            cursor += 1;
        }
    }
    projected
}

/// Construye la distribución previa p0. Si no se especifica, usa uniforme sobre posiciones permitidas.
fn build_prior(
    prior: Option<&Vec<f64>>,
    number_of_keys: usize,
    allowed: &[bool],
) -> Result<Vec<f64>> {
    let mut values = match prior {
        Some(values) => values.clone(),
        None => vec![1.0; number_of_keys],
    };

    anyhow::ensure!(
        values.len() == number_of_keys,
        "prior debe tener la misma cantidad de entradas que keys."
    );
    anyhow::ensure!(
        values.iter().all(|v| *v >= 0.0),
        "prior debe tener entradas no negativas."
    );

    for (value, is_allowed) in values.iter_mut().zip(allowed) {
        if !*is_allowed {
            *value = 0.0;
        }
    }

    let total: f64 = values.iter().sum();
    anyhow::ensure!(
        total > 0.0,
        "prior no tiene masa positiva en las posiciones permitidas por la máscara."
    );

    Ok(values.iter().map(|v| v / total).collect())
}

fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let eps = 1.0e-15;
    p.iter()
        .zip(q)
        .filter(|(pi, _)| **pi > 0.0)
        .map(|(pi, qi)| pi * (pi.max(eps) / qi.max(eps)).ln())
        .sum()
}

fn jensen_shannon_distance(p: &[f64], q: &[f64]) -> f64 {
    let midpoint: Vec<f64> = p.iter().zip(q).map(|(a, b)| 0.5 * (a + b)).collect();
    let js = 0.5 * kl_divergence(p, &midpoint) + 0.5 * kl_divergence(q, &midpoint);
    js.max(0.0).sqrt()
}

fn top_k_mass(probabilities: &[f64], k: usize) -> f64 {
    let mut sorted = probabilities.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted.iter().take(k.min(sorted.len())).sum()
}

fn l1_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

fn average_outputs(heads: &[HeadAttentionResult]) -> Vec<f64> {
    let out_dim = heads[0].regularized_output.len();
    let mut output = vec![0.0; out_dim];
    for head in heads {
        for (j, value) in output.iter_mut().enumerate().take(out_dim) {
            *value += head.regularized_output[j] / heads.len() as f64;
        }
    }
    output
}

fn pairwise_head_metrics(heads: &[HeadAttentionResult]) -> (f64, f64, f64) {
    if heads.len() < 2 {
        return (0.0, 0.0, 0.0);
    }

    let mut count = 0.0;
    let mut sum_l1 = 0.0;
    let mut sum_l2 = 0.0;
    let mut sum_js = 0.0;

    for i in 0..heads.len() {
        for j in (i + 1)..heads.len() {
            let diff = sub(&heads[i].regularized_weights, &heads[j].regularized_weights);
            sum_l1 += diff.iter().map(|v| v.abs()).sum::<f64>();
            sum_l2 += norm2(&diff);
            sum_js += jensen_shannon_distance(
                &heads[i].regularized_weights,
                &heads[j].regularized_weights,
            );
            count += 1.0;
        }
    }

    (sum_l1 / count, sum_l2 / count, sum_js / count)
}

fn mean<I>(values: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0.0;
    for value in values {
        sum += value;
        count += 1.0;
    }
    if count == 0.0 {
        0.0
    } else {
        sum / count
    }
}

fn score_candidate(
    objective: &AgentObjective,
    entropy: f64,
    distance_to_prior: f64,
    difference_from_softmax: f64,
    output_shift: f64,
    js_softmax_regularized: f64,
) -> f64 {
    match objective {
        AgentObjective::MaxEntropy => entropy,
        AgentObjective::MinDistanceToPrior => -distance_to_prior,
        AgentObjective::MaxDifferenceFromSoftmax => difference_from_softmax,
        AgentObjective::MinOutputShift => -output_shift,
        AgentObjective::BalancedTradeoff => {
            entropy + 0.5 * js_softmax_regularized - 0.5 * distance_to_prior - 0.25 * output_shift
        }
    }
}

fn format_vector_special(x: &[f64]) -> String {
    let values: Vec<String> = x
        .iter()
        .map(|v| {
            if v.is_infinite() && v.is_sign_negative() {
                "-inf".to_string()
            } else if v.is_infinite() && v.is_sign_positive() {
                "inf".to_string()
            } else {
                format!("{:.10}", v)
            }
        })
        .collect();
    format!("[{}]", values.join(","))
}

#[cfg(test)]
mod tests {
    use super::{project_to_masked_simplex, project_to_simplex, softmax};

    #[test]
    fn projection_sums_to_one() {
        let projected = project_to_simplex(&[0.2, -1.0, 2.0]);
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-9);
        assert!(projected.iter().all(|v| *v >= 0.0));
    }

    #[test]
    fn masked_projection_respects_blocked_positions() {
        let projected = project_to_masked_simplex(&[0.2, 0.4, 0.4], &[true, false, true]);
        assert_eq!(projected[1], 0.0);
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn softmax_gives_zero_to_negative_infinity() {
        let weights = softmax(&[1.0, f64::NEG_INFINITY, 2.0]);
        assert_eq!(weights[1], 0.0);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-9);
    }
}
