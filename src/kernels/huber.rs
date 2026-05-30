use crate::kernels::KernelFunction;
use anyhow::Result;

/// Kernel Huber separable: weight * sum_i h_delta(x_i).
pub struct HuberKernel {
    delta: f64,
    weight: f64,
}

impl HuberKernel {
    pub fn new(delta: f64, weight: Option<f64>) -> Result<Self> {
        anyhow::ensure!(delta > 0.0, "delta debe ser positivo.");
        let weight = weight.unwrap_or(1.0);
        anyhow::ensure!(weight >= 0.0, "weight debe ser no negativo.");
        Ok(Self { delta, weight })
    }

    fn scalar_value(&self, value: f64) -> f64 {
        let abs = value.abs();
        if abs <= self.delta {
            0.5 * value * value
        } else {
            self.delta * (abs - 0.5 * self.delta)
        }
    }

    fn scalar_grad(&self, value: f64) -> f64 {
        if value.abs() <= self.delta {
            value
        } else {
            self.delta * value.signum()
        }
    }
}

impl KernelFunction for HuberKernel {
    fn name(&self) -> &'static str {
        "huber"
    }

    fn value(&self, x: &[f64]) -> f64 {
        self.weight * x.iter().map(|v| self.scalar_value(*v)).sum::<f64>()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .map(|v| self.weight * self.scalar_grad(*v))
            .collect()
    }
}
