use crate::config::FunctionConfig;
use crate::functions::{build_conjugate_function, build_function};
use crate::math::{dot, norm2_squared, sub};
use crate::optimization::projections::{project_to_box, project_to_simplex};
use anyhow::{Context, Result};
use serde_yaml;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ProxResult {
    pub function_name: String,
    pub point: Vec<f64>,
    pub step: f64,
    pub prox: Vec<f64>,
    pub function_value_at_prox: f64,
    pub moreau_value: f64,
    pub moreau_gradient: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct FenchelYoungResult {
    pub function_name: String,
    pub x: Vec<f64>,
    pub s: Vec<f64>,
    pub f_value: f64,
    pub conjugate_value: f64,
    pub pairing: f64,
    pub gap: f64,
    pub relative_gap: f64,
    pub passed: bool,
}

pub fn load_function_config(path: &Path) -> Result<FunctionConfig> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("No se pudo leer la función: {}", path.display()))?;
    serde_yaml::from_str(&text)
        .with_context(|| format!("No se pudo parsear función YAML: {}", path.display()))
}

pub fn parse_vector(text: &str) -> Result<Vec<f64>> {
    let values: Vec<f64> =
        serde_yaml::from_str(text).with_context(|| format!("No se pudo parsear vector: {text}"))?;
    anyhow::ensure!(!values.is_empty(), "El vector no puede estar vacío.");
    anyhow::ensure!(
        values.iter().all(|value| value.is_finite()),
        "El vector debe contener solo valores finitos."
    );
    Ok(values)
}

pub fn compute_prox(config: &FunctionConfig, point: &[f64], step: f64) -> Result<ProxResult> {
    anyhow::ensure!(step > 0.0, "--step debe ser positivo.");
    anyhow::ensure!(!point.is_empty(), "El punto no puede estar vacío.");

    let prox = proximal_point(config, point, step)?;
    let function = build_function(config)?;
    let function_value_at_prox = function.value(&prox);
    let displacement = sub(point, &prox);
    let moreau_value = function_value_at_prox + norm2_squared(&displacement) / (2.0 * step);
    let moreau_gradient = displacement.iter().map(|value| value / step).collect();

    Ok(ProxResult {
        function_name: function.name().to_string(),
        point: point.to_vec(),
        step,
        prox,
        function_value_at_prox,
        moreau_value,
        moreau_gradient,
    })
}

pub fn check_fenchel_young(
    config: &FunctionConfig,
    x: &[f64],
    s: &[f64],
    tolerance: f64,
) -> Result<FenchelYoungResult> {
    anyhow::ensure!(
        x.len() == s.len(),
        "x y s deben tener la misma dimensión para Fenchel-Young."
    );
    anyhow::ensure!(tolerance > 0.0, "--tolerance debe ser positiva.");

    let function = build_function(config)?;
    let conjugate = build_conjugate_function(config)?;
    let f_value = function.value(x);
    let conjugate_value = conjugate.value(s);
    let pairing = dot(x, s);
    let gap = f_value + conjugate_value - pairing;
    let relative_gap = if gap.is_finite() {
        gap.abs() / (1.0 + pairing.abs())
    } else {
        f64::INFINITY
    };
    let passed = if gap.is_infinite() && gap.is_sign_positive() {
        true
    } else {
        gap >= -tolerance
    };

    Ok(FenchelYoungResult {
        function_name: function.name().to_string(),
        x: x.to_vec(),
        s: s.to_vec(),
        f_value,
        conjugate_value,
        pairing,
        gap,
        relative_gap,
        passed,
    })
}

pub fn proximal_point(config: &FunctionConfig, point: &[f64], step: f64) -> Result<Vec<f64>> {
    match config {
        FunctionConfig::L1 { alpha } => Ok(soft_threshold(point, step * *alpha)),
        FunctionConfig::L2 { alpha } => {
            let scale = 1.0 / (1.0 + step * *alpha);
            Ok(point.iter().map(|value| value * scale).collect())
        }
        FunctionConfig::ElasticNet { l1_alpha, l2_alpha } => {
            let thresholded = soft_threshold(point, step * *l1_alpha);
            let scale = 1.0 / (1.0 + step * *l2_alpha);
            Ok(thresholded.iter().map(|value| value * scale).collect())
        }
        FunctionConfig::IndicatorBox { lower, upper } => {
            anyhow::ensure!(
                point.len() == lower.len() && point.len() == upper.len(),
                "El punto debe tener la misma dimensión que la caja."
            );
            Ok(project_to_box(point, lower, upper))
        }
        FunctionConfig::IndicatorSimplex { .. } => Ok(project_to_simplex(point)),
        FunctionConfig::Quadratic { .. }
        | FunctionConfig::Huber { .. }
        | FunctionConfig::HingeLoss { .. }
        | FunctionConfig::LogisticLoss { .. }
        | FunctionConfig::MaxAffine { .. } => anyhow::bail!(
            "prox todavía no está implementado para esta función. Use l1, l2, elastic-net, indicator-box o indicator-simplex."
        ),
    }
}

fn soft_threshold(values: &[f64], threshold: f64) -> Vec<f64> {
    values
        .iter()
        .map(|value| {
            if *value > threshold {
                value - threshold
            } else if *value < -threshold {
                value + threshold
            } else {
                0.0
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prox_l1_uses_soft_thresholding() {
        let config = FunctionConfig::L1 { alpha: 1.0 };
        let result = compute_prox(&config, &[1.0, -2.0, 0.05], 0.1).unwrap();
        assert!((result.prox[0] - 0.9).abs() < 1.0e-12);
        assert!((result.prox[1] + 1.9).abs() < 1.0e-12);
        assert_eq!(result.prox[2], 0.0);
    }

    #[test]
    fn fenchel_young_l1_passes_inside_linf_ball() {
        let config = FunctionConfig::L1 { alpha: 1.0 };
        let result = check_fenchel_young(&config, &[1.0, -2.0], &[0.5, -0.25], 1.0e-9).unwrap();
        assert!(result.passed);
        assert!(result.gap >= -1.0e-9);
    }
}
