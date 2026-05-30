use crate::functions::ConvexFunction;
use anyhow::Result;

/// Indicador del simplex probabilístico: x_i >= 0 y sum_i x_i = 1.
pub struct IndicatorSimplexFunction {
    tolerance: f64,
}

impl IndicatorSimplexFunction {
    pub fn new(tolerance: Option<f64>) -> Result<Self> {
        let tolerance = tolerance.unwrap_or(1.0e-8);
        anyhow::ensure!(tolerance > 0.0, "tolerance debe ser positiva.");
        Ok(Self { tolerance })
    }
}

impl ConvexFunction for IndicatorSimplexFunction {
    fn name(&self) -> &'static str {
        "indicator-simplex"
    }

    fn value(&self, x: &[f64]) -> f64 {
        let sum: f64 = x.iter().sum();
        let nonnegative = x.iter().all(|v| *v >= -self.tolerance);
        if nonnegative && (sum - 1.0).abs() <= self.tolerance {
            0.0
        } else {
            f64::INFINITY
        }
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        // Dentro del simplex elegimos el vector cero como normal compatible.
        vec![0.0; x.len()]
    }

    fn simplex_constraint(&self) -> bool {
        true
    }
}
