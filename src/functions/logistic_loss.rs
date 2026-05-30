use crate::functions::ConvexFunction;
use crate::math::{dot, norm2_squared};
use anyhow::Result;

/// Pérdida logística lineal: weight * sum log(1+exp(-y_i<a_i,x>)) + l2_alpha/2 ||x||².
pub struct LogisticLossFunction {
    samples: Vec<Vec<f64>>,
    labels: Vec<f64>,
    l2_alpha: f64,
    weight: f64,
}

impl LogisticLossFunction {
    pub fn new(
        samples: Vec<Vec<f64>>,
        labels: Vec<f64>,
        l2_alpha: Option<f64>,
        weight: Option<f64>,
    ) -> Result<Self> {
        anyhow::ensure!(!samples.is_empty(), "samples no puede estar vacío.");
        anyhow::ensure!(
            samples.len() == labels.len(),
            "samples y labels deben tener la misma longitud."
        );
        let dimension = samples[0].len();
        anyhow::ensure!(
            samples.iter().all(|sample| sample.len() == dimension),
            "todos los samples deben tener la misma dimensión."
        );
        anyhow::ensure!(
            labels
                .iter()
                .all(|value| (*value - 1.0).abs() <= 1.0e-12 || (*value + 1.0).abs() <= 1.0e-12),
            "labels debe contener solo -1 o 1."
        );
        let l2_alpha = l2_alpha.unwrap_or(0.0);
        let weight = weight.unwrap_or(1.0);
        anyhow::ensure!(l2_alpha >= 0.0, "l2_alpha debe ser no negativo.");
        anyhow::ensure!(weight >= 0.0, "weight debe ser no negativo.");
        Ok(Self {
            samples,
            labels,
            l2_alpha,
            weight,
        })
    }

    fn softplus(value: f64) -> f64 {
        if value > 0.0 {
            value + (-value).exp().ln_1p()
        } else {
            value.exp().ln_1p()
        }
    }
}

impl ConvexFunction for LogisticLossFunction {
    fn name(&self) -> &'static str {
        "logistic-loss"
    }

    fn value(&self, x: &[f64]) -> f64 {
        let data_loss = self
            .samples
            .iter()
            .zip(&self.labels)
            .map(|(sample, label)| Self::softplus(-label * dot(sample, x)))
            .sum::<f64>();
        self.weight * data_loss + 0.5 * self.l2_alpha * norm2_squared(x)
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        let mut grad: Vec<f64> = x.iter().map(|v| self.l2_alpha * v).collect();
        for (sample, label) in self.samples.iter().zip(&self.labels) {
            let margin = label * dot(sample, x);
            let coeff = if margin >= 0.0 {
                -label * (-margin).exp() / (1.0 + (-margin).exp())
            } else {
                -label / (1.0 + margin.exp())
            };
            for i in 0..grad.len() {
                grad[i] += self.weight * coeff * sample[i];
            }
        }
        grad
    }

}
