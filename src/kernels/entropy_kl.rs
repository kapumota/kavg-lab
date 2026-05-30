use crate::kernels::KernelFunction;
use anyhow::Result;

/// Kernel KL desplazado: sum_i x_i log(x_i/r_i) - x_i + r_i.
/// Se usa epsilon para evitar log(0) en experimentos numéricos pequeños.
pub struct EntropyKlKernel {
    reference: Option<Vec<f64>>,
    epsilon: f64,
}

impl EntropyKlKernel {
    pub fn new(reference: Option<Vec<f64>>, epsilon: Option<f64>) -> Result<Self> {
        let epsilon = epsilon.unwrap_or(1.0e-12);
        anyhow::ensure!(epsilon > 0.0, "epsilon debe ser positivo.");
        if let Some(reference) = &reference {
            anyhow::ensure!(
                reference.iter().all(|v| *v > 0.0),
                "reference debe contener valores positivos."
            );
        }
        Ok(Self { reference, epsilon })
    }

    fn reference_at(&self, index: usize) -> f64 {
        self.reference
            .as_ref()
            .and_then(|values| values.get(index).copied())
            .unwrap_or(1.0)
    }
}

impl KernelFunction for EntropyKlKernel {
    fn name(&self) -> &'static str {
        "entropy-kl"
    }

    fn value(&self, x: &[f64]) -> f64 {
        x.iter()
            .enumerate()
            .map(|(i, value)| {
                let xi = value.max(self.epsilon);
                let ri = self.reference_at(i);
                xi * (xi / ri).ln() - xi + ri
            })
            .sum()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, value)| {
                let xi = value.max(self.epsilon);
                let ri = self.reference_at(i);
                (xi / ri).ln()
            })
            .collect()
    }
}
