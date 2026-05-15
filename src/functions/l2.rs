use crate::functions::{ConvexFunction, QuadraticForm};
use crate::math::norm2_squared;
use anyhow::Result;

/// Función L2 suave de la forma f(x) = (alpha/2) ||x||².
pub struct L2Function {
    alpha: f64,
}

impl L2Function {
    pub fn new(alpha: f64) -> Result<Self> {
        anyhow::ensure!(alpha >= 0.0, "alpha debe ser no negativo.");
        Ok(Self { alpha })
    }
}

impl ConvexFunction for L2Function {
    fn name(&self) -> &'static str {
        "l2"
    }

    fn value(&self, x: &[f64]) -> f64 {
        0.5 * self.alpha * norm2_squared(x)
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        // Como la función es diferenciable, el subgradiente coincide con el gradiente.
        x.iter().map(|v| self.alpha * v).collect()
    }

    fn quadratic_form(&self, dimension: usize) -> Option<QuadraticForm> {
        let mut hessian = vec![vec![0.0; dimension]; dimension];
        for (i, row) in hessian.iter_mut().enumerate().take(dimension) {
            row[i] = self.alpha;
        }
        Some(QuadraticForm {
            hessian,
            linear: vec![0.0; dimension],
            constant: 0.0,
        })
    }
}
