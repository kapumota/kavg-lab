use crate::functions::ConvexFunction;
use crate::math::dot;
use anyhow::Result;

/// Pérdida hinge para clasificación lineal: weight * sum max(0, 1 - y_i <a_i,x>).
pub struct HingeLossFunction {
    samples: Vec<Vec<f64>>,
    labels: Vec<f64>,
    weight: f64,
}

impl HingeLossFunction {
    pub fn new(samples: Vec<Vec<f64>>, labels: Vec<f64>, weight: Option<f64>) -> Result<Self> {
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
        let weight = weight.unwrap_or(1.0);
        anyhow::ensure!(weight >= 0.0, "weight debe ser no negativo.");
        Ok(Self {
            samples,
            labels,
            weight,
        })
    }
}

impl ConvexFunction for HingeLossFunction {
    fn name(&self) -> &'static str {
        "hinge-loss"
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.weight
            * self
                .samples
                .iter()
                .zip(&self.labels)
                .map(|(sample, label)| (1.0 - label * dot(sample, x)).max(0.0))
                .sum::<f64>()
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        let mut grad = vec![0.0; x.len()];
        for (sample, label) in self.samples.iter().zip(&self.labels) {
            if 1.0 - label * dot(sample, x) > 0.0 {
                for i in 0..grad.len() {
                    grad[i] -= self.weight * label * sample[i];
                }
            }
        }
        grad
    }
}
