use crate::functions::{ConvexFunction, QuadraticForm};
use crate::math::norm2_squared;
use anyhow::Result;

/// Elastic net: f(x) = l1_alpha ||x||₁ + (l2_alpha/2)||x||².
pub struct ElasticNetFunction {
    l1_alpha: f64,
    l2_alpha: f64,
}

impl ElasticNetFunction {
    pub fn new(l1_alpha: f64, l2_alpha: f64) -> Result<Self> {
        anyhow::ensure!(l1_alpha >= 0.0, "l1_alpha debe ser no negativo.");
        anyhow::ensure!(l2_alpha >= 0.0, "l2_alpha debe ser no negativo.");
        Ok(Self { l1_alpha, l2_alpha })
    }
}

impl ConvexFunction for ElasticNetFunction {
    fn name(&self) -> &'static str {
        "elastic-net"
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.l1_alpha * x.iter().map(|v| v.abs()).sum::<f64>()
            + 0.5 * self.l2_alpha * norm2_squared(x)
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|v| {
                let l1_part = if *v > 0.0 {
                    self.l1_alpha
                } else if *v < 0.0 {
                    -self.l1_alpha
                } else {
                    0.0
                };
                l1_part + self.l2_alpha * v
            })
            .collect()
    }

    fn quadratic_form(&self, dimension: usize) -> Option<QuadraticForm> {
        let mut hessian = vec![vec![0.0; dimension]; dimension];
        for (i, row) in hessian.iter_mut().enumerate().take(dimension) {
            row[i] = self.l2_alpha;
        }
        Some(QuadraticForm {
            hessian,
            linear: vec![0.0; dimension],
            constant: 0.0,
        })
    }

    fn l1_alpha(&self) -> Option<f64> {
        Some(self.l1_alpha)
    }
}
