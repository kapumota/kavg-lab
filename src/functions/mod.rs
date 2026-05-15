mod conjugates;
mod l1;
mod l2;
mod quadratic;

use crate::config::FunctionConfig;
use anyhow::Result;
pub use conjugates::{build_conjugate_function, L2ConjugateFunction, QuadraticConjugateFunction};
pub use l1::L1Function;
pub use l2::L2Function;
pub use quadratic::QuadraticFunction;

/// Forma cuadrática densa: f(x) = 1/2 xᵀ H x + qᵀ x + c.
/// Esta representación se usa para transformar el kernel average a un QP.
#[derive(Debug, Clone)]
pub struct QuadraticForm {
    pub hessian: Vec<Vec<f64>>,
    pub linear: Vec<f64>,
    pub constant: f64,
}

/// Interfaz común para funciones convexas.
/// Los nombres de métodos están en inglés; los comentarios están en español.
pub trait ConvexFunction {
    fn name(&self) -> &'static str;
    fn value(&self, x: &[f64]) -> f64;
    fn subgradient(&self, x: &[f64]) -> Vec<f64>;

    /// Devuelve una forma cuadrática si la función puede escribirse como
    /// 1/2 xᵀ H x + qᵀ x + c. El backend OSQP usa esta información.
    fn quadratic_form(&self, _dimension: usize) -> Option<QuadraticForm> {
        None
    }

    /// Devuelve alpha si la función es alpha ||x||₁.
    /// El backend OSQP usa variables auxiliares para linealizar el valor absoluto.
    fn l1_alpha(&self) -> Option<f64> {
        None
    }
}

pub fn build_function(config: &FunctionConfig) -> Result<Box<dyn ConvexFunction>> {
    match config {
        FunctionConfig::Quadratic { matrix, vector } => Ok(Box::new(QuadraticFunction::new(
            matrix.clone(),
            vector.clone(),
        )?)),
        FunctionConfig::L1 { alpha } => Ok(Box::new(L1Function::new(*alpha)?)),
        FunctionConfig::L2 { alpha } => Ok(Box::new(L2Function::new(*alpha)?)),
    }
}
