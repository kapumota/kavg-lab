use crate::functions::ConvexFunction;
use anyhow::Result;

/// Función L1 de la forma f(x) = alpha ||x||₁.
pub struct L1Function {
    alpha: f64,
}

impl L1Function {
    pub fn new(alpha: f64) -> Result<Self> {
        anyhow::ensure!(alpha >= 0.0, "alpha debe ser no negativo.");
        Ok(Self { alpha })
    }
}

impl ConvexFunction for L1Function {
    fn name(&self) -> &'static str {
        "l1"
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.alpha * x.iter().map(|v| v.abs()).sum::<f64>()
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        // En cero elegimos el subgradiente 0, que pertenece al intervalo [-alpha, alpha].
        x.iter()
            .map(|v| {
                if *v > 0.0 {
                    self.alpha
                } else if *v < 0.0 {
                    -self.alpha
                } else {
                    0.0
                }
            })
            .collect()
    }

    fn l1_alpha(&self) -> Option<f64> {
        Some(self.alpha)
    }
}
