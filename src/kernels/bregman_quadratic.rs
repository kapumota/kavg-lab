use crate::kernels::{KernelFunction, KernelQuadraticForm};
use crate::math::{dot, mat_vec, sub};
use anyhow::Result;

/// Divergencia de Bregman cuadrática: g(x)=1/2 (x-c)ᵀM(x-c).
pub struct BregmanQuadraticKernel {
    matrix: Vec<Vec<f64>>,
    center: Vec<f64>,
}

impl BregmanQuadraticKernel {
    pub fn new(matrix: Vec<Vec<f64>>, center: Option<Vec<f64>>) -> Result<Self> {
        anyhow::ensure!(!matrix.is_empty(), "matrix no puede estar vacía.");
        let dimension = matrix.len();
        anyhow::ensure!(
            matrix.iter().all(|row| row.len() == dimension),
            "matrix debe ser cuadrada."
        );
        let center = center.unwrap_or_else(|| vec![0.0; dimension]);
        anyhow::ensure!(
            center.len() == dimension,
            "center debe tener la misma dimensión que matrix."
        );
        Ok(Self { matrix, center })
    }
}

impl KernelFunction for BregmanQuadraticKernel {
    fn name(&self) -> &'static str {
        "bregman-quadratic"
    }

    fn value(&self, x: &[f64]) -> f64 {
        let shifted = sub(x, &self.center);
        let m_shifted = mat_vec(&self.matrix, &shifted);
        0.5 * dot(&shifted, &m_shifted)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        let shifted = sub(x, &self.center);
        mat_vec(&self.matrix, &shifted)
    }

    fn quadratic_form(&self, dimension: usize) -> Option<KernelQuadraticForm> {
        if self.matrix.len() != dimension || self.center.len() != dimension {
            return None;
        }
        let m_center = mat_vec(&self.matrix, &self.center);
        let constant = 0.5 * dot(&self.center, &m_center);
        Some(KernelQuadraticForm {
            hessian: self.matrix.clone(),
            linear: m_center.iter().map(|v| -v).collect(),
            constant,
        })
    }
}
