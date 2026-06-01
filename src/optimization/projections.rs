/// Proyección euclidiana al simplex {x >= 0, sum x = 1}.
///
/// Implementa el algoritmo de ordenamiento estándar para la proyección sobre el simplex.
pub fn project_to_simplex(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

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

/// Proyección al simplex respetando una máscara booleana.
pub fn project_to_masked_simplex(values: &[f64], allowed: &[bool]) -> Vec<f64> {
    let allowed_values: Vec<f64> = values
        .iter()
        .zip(allowed)
        .filter_map(|(value, is_allowed)| if *is_allowed { Some(*value) } else { None })
        .collect();

    if allowed_values.is_empty() {
        return vec![0.0; values.len()];
    }

    let projected_allowed = project_to_simplex(&allowed_values);
    scatter_allowed(values.len(), allowed, &projected_allowed)
}

/// Sparsemax como proyección de scores al simplex.
pub fn sparsemax(scores: &[f64]) -> Vec<f64> {
    let finite_scores = sanitize_scores(scores);
    project_to_simplex(&finite_scores)
}

/// Sparsemax con máscara estructurada.
pub fn masked_sparsemax(scores: &[f64], allowed: &[bool]) -> Vec<f64> {
    let values: Vec<f64> = scores
        .iter()
        .zip(allowed)
        .map(|(score, is_allowed)| {
            if *is_allowed && score.is_finite() {
                *score
            } else {
                0.0
            }
        })
        .collect();
    project_to_masked_simplex(&values, allowed)
}

/// Top-k sparse attention: conserva los k scores permitidos más altos y proyecta al simplex.
pub fn top_k_masked_sparsemax(scores: &[f64], allowed: &[bool], k: usize) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .filter(|(index, score)| allowed[*index] && score.is_finite())
        .map(|(index, score)| (index, *score))
        .collect();

    if indexed.is_empty() {
        return vec![0.0; scores.len()];
    }

    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep = k.max(1).min(indexed.len());
    let mut top_allowed = vec![false; scores.len()];
    for (index, _) in indexed.into_iter().take(keep) {
        top_allowed[index] = true;
    }
    masked_sparsemax(scores, &top_allowed)
}

/// Entmax 1.5 aproximado mediante búsqueda de umbral.
///
/// Esta implementación es determinista y suficiente para experimentación CLI.
pub fn masked_entmax15(scores: &[f64], allowed: &[bool]) -> Vec<f64> {
    let active: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .filter(|(index, score)| allowed[*index] && score.is_finite())
        .map(|(index, score)| (index, *score))
        .collect();

    if active.is_empty() {
        return vec![0.0; scores.len()];
    }

    let max_score = active
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);
    let shifted: Vec<(usize, f64)> = active
        .into_iter()
        .map(|(index, score)| (index, score - max_score))
        .collect();

    let mut lower = shifted
        .iter()
        .map(|(_, score)| *score)
        .fold(f64::INFINITY, f64::min)
        - 2.0;
    let mut upper = 0.0;

    for _ in 0..80 {
        let tau = 0.5 * (lower + upper);
        let mass: f64 = shifted
            .iter()
            .map(|(_, score)| ((score - tau) / 2.0).max(0.0).powi(2))
            .sum();
        if mass > 1.0 {
            lower = tau;
        } else {
            upper = tau;
        }
    }

    let tau = 0.5 * (lower + upper);
    let mut output = vec![0.0; scores.len()];
    for (index, score) in shifted {
        output[index] = ((score - tau) / 2.0).max(0.0).powi(2);
    }
    normalize_nonnegative(&mut output);
    output
}

/// Proyección de Dykstra para intersecar una caja [lower, upper] con el simplex.
pub fn dykstra_project_box_simplex(
    values: &[f64],
    lower: &[f64],
    upper: &[f64],
    iterations: usize,
) -> Vec<f64> {
    assert_eq!(values.len(), lower.len());
    assert_eq!(values.len(), upper.len());

    let mut x = values.to_vec();
    let mut p = vec![0.0; values.len()];
    let mut q = vec![0.0; values.len()];

    for _ in 0..iterations.max(1) {
        let y: Vec<f64> = x.iter().zip(&p).map(|(xi, pi)| xi + pi).collect();
        let projected_box = project_to_box(&y, lower, upper);
        p = y
            .iter()
            .zip(&projected_box)
            .map(|(yi, zi)| yi - zi)
            .collect();

        let y: Vec<f64> = projected_box
            .iter()
            .zip(&q)
            .map(|(xi, qi)| xi + qi)
            .collect();
        let projected_simplex = project_to_simplex(&y);
        q = y
            .iter()
            .zip(&projected_simplex)
            .map(|(yi, zi)| yi - zi)
            .collect();
        x = projected_simplex;
    }

    x
}

pub fn project_to_box(values: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    values
        .iter()
        .zip(lower)
        .zip(upper)
        .map(|((value, lo), hi)| value.max(*lo).min(*hi))
        .collect()
}

fn sanitize_scores(scores: &[f64]) -> Vec<f64> {
    let finite: Vec<f64> = scores
        .iter()
        .copied()
        .filter(|score| score.is_finite())
        .collect();
    if finite.is_empty() {
        return vec![0.0; scores.len()];
    }
    let min_finite = finite.iter().copied().fold(f64::INFINITY, f64::min);
    scores
        .iter()
        .map(|score| {
            if score.is_finite() {
                *score
            } else {
                min_finite - 1.0
            }
        })
        .collect()
}

fn scatter_allowed(length: usize, allowed: &[bool], projected_allowed: &[f64]) -> Vec<f64> {
    let mut projected = vec![0.0; length];
    let mut cursor = 0;
    for (index, is_allowed) in allowed.iter().enumerate() {
        if *is_allowed {
            projected[index] = projected_allowed[cursor];
            cursor += 1;
        }
    }
    projected
}

fn normalize_nonnegative(values: &mut [f64]) {
    for value in values.iter_mut() {
        if *value < 0.0 || !value.is_finite() {
            *value = 0.0;
        }
    }
    let total: f64 = values.iter().sum();
    if total > 0.0 {
        for value in values.iter_mut() {
            *value /= total;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simplex_projection_sums_to_one() {
        let projected = project_to_simplex(&[0.2, -1.0, 2.0]);
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-9);
        assert!(projected.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn sparsemax_can_return_exact_zeros() {
        let weights = sparsemax(&[3.0, 1.0, -2.0]);
        assert_eq!(weights[2], 0.0);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn top_k_sparsemax_keeps_only_k_entries() {
        let weights = top_k_masked_sparsemax(&[3.0, 2.0, 1.0, 0.0], &[true; 4], 2);
        assert!(weights.iter().filter(|value| **value > 0.0).count() <= 2);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn entmax15_is_normalized() {
        let weights = masked_entmax15(&[1.0, 0.5, -1.0], &[true, true, true]);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-8);
        assert!(weights.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn dykstra_box_simplex_respects_constraints() {
        let projected =
            dykstra_project_box_simplex(&[2.0, -1.0, 0.5], &[0.0, 0.0, 0.0], &[0.8, 0.8, 0.8], 50);
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!(projected
            .iter()
            .all(|value| *value >= -1.0e-9 && *value <= 0.8 + 1.0e-9));
    }
}
