use crate::kernels::{KernelFunction, KernelQuadraticForm};
use crate::math::{dot, mat_vec};
use anyhow::Result;

/// Kernel de Mahalanobis: g(x)=1/2 xᵀMx.
pub struct MahalanobisKernel {
    matrix: Vec<Vec<f64>>,
}

impl MahalanobisKernel {
    pub fn new(matrix: Vec<Vec<f64>>) -> Result<Self> {
        anyhow::ensure!(!matrix.is_empty(), "matrix no puede estar vacía.");
        let dimension = matrix.len();
        anyhow::ensure!(
            matrix.iter().all(|row| row.len() == dimension),
            "matrix debe ser cuadrada."
        );
        anyhow::ensure!(
            matrix.iter().enumerate().all(|(i, row)| row[i] >= 0.0),
            "matrix debe tener diagonal no negativa."
        );
        Ok(Self { matrix })
    }
}

impl KernelFunction for MahalanobisKernel {
    fn name(&self) -> &'static str {
        "mahalanobis"
    }

    fn value(&self, x: &[f64]) -> f64 {
        let mx = mat_vec(&self.matrix, x);
        0.5 * dot(x, &mx)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        mat_vec(&self.matrix, x)
    }

    fn quadratic_form(&self, dimension: usize) -> Option<KernelQuadraticForm> {
        if self.matrix.len() != dimension {
            return None;
        }
        Some(KernelQuadraticForm {
            hessian: self.matrix.clone(),
            linear: vec![0.0; dimension],
            constant: 0.0,
        })
    }
}
