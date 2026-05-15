use crate::functions::{ConvexFunction, QuadraticForm};
use crate::math::{mat_t_vec, mat_vec, norm2_squared, sub};
use anyhow::Result;

/// Función cuadrática de la forma f(x) = 1/2 ||Ax - b||².
pub struct QuadraticFunction {
    matrix: Vec<Vec<f64>>,
    vector: Vec<f64>,
}

impl QuadraticFunction {
    pub fn new(matrix: Vec<Vec<f64>>, vector: Vec<f64>) -> Result<Self> {
        anyhow::ensure!(!matrix.is_empty(), "La matriz no puede estar vacía.");
        let cols = matrix[0].len();
        anyhow::ensure!(cols > 0, "La matriz debe tener al menos una columna.");
        anyhow::ensure!(
            matrix.iter().all(|row| row.len() == cols),
            "Todas las filas de la matriz deben tener la misma longitud."
        );
        anyhow::ensure!(
            matrix.len() == vector.len(),
            "La dimensión de b debe coincidir con el número de filas de A."
        );
        Ok(Self { matrix, vector })
    }

    fn hessian(&self) -> Vec<Vec<f64>> {
        let cols = self.matrix[0].len();
        let mut out = vec![vec![0.0; cols]; cols];
        for row in &self.matrix {
            for i in 0..cols {
                for j in 0..cols {
                    out[i][j] += row[i] * row[j];
                }
            }
        }
        out
    }
}

impl ConvexFunction for QuadraticFunction {
    fn name(&self) -> &'static str {
        "quadratic"
    }

    fn value(&self, x: &[f64]) -> f64 {
        let ax = mat_vec(&self.matrix, x);
        let residual = sub(&ax, &self.vector);
        0.5 * norm2_squared(&residual)
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        // Para una cuadrática suave, el subgradiente coincide con el gradiente: A^T(Ax-b).
        let ax = mat_vec(&self.matrix, x);
        let residual = sub(&ax, &self.vector);
        mat_t_vec(&self.matrix, &residual)
    }

    fn quadratic_form(&self, dimension: usize) -> Option<QuadraticForm> {
        if self.matrix[0].len() != dimension {
            return None;
        }
        let hessian = self.hessian();
        let at_b = mat_t_vec(&self.matrix, &self.vector);
        let linear: Vec<f64> = at_b.into_iter().map(|v| -v).collect();
        let constant = 0.5 * norm2_squared(&self.vector);
        Some(QuadraticForm {
            hessian,
            linear,
            constant,
        })
    }
}
