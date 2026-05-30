use crate::kernels::{KernelFunction, KernelQuadraticForm};
use anyhow::Result;

/// Kernel diagonal: g(x)=1/2 sum_i w_i x_i².
pub struct WeightedSquaredNormKernel {
    weights: Vec<f64>,
}

impl WeightedSquaredNormKernel {
    pub fn new(weights: Vec<f64>) -> Result<Self> {
        anyhow::ensure!(!weights.is_empty(), "weights no puede estar vacío.");
        anyhow::ensure!(
            weights.iter().all(|w| *w >= 0.0),
            "weights debe contener valores no negativos."
        );
        Ok(Self { weights })
    }
}

impl KernelFunction for WeightedSquaredNormKernel {
    fn name(&self) -> &'static str {
        "weighted-squared-norm"
    }

    fn value(&self, x: &[f64]) -> f64 {
        0.5 * x
            .iter()
            .zip(&self.weights)
            .map(|(v, w)| w * v * v)
            .sum::<f64>()
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.weights).map(|(v, w)| w * v).collect()
    }

    fn quadratic_form(&self, dimension: usize) -> Option<KernelQuadraticForm> {
        if self.weights.len() != dimension {
            return None;
        }
        let mut hessian = vec![vec![0.0; dimension]; dimension];
        for (i, row) in hessian.iter_mut().enumerate().take(dimension) {
            row[i] = self.weights[i];
        }
        Some(KernelQuadraticForm {
            hessian,
            linear: vec![0.0; dimension],
            constant: 0.0,
        })
    }
}
