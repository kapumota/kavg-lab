mod conjugates;
mod elastic_net;
mod hinge_loss;
mod huber;
mod indicator_box;
mod indicator_simplex;
mod l1;
mod l2;
mod logistic_loss;
mod max_affine;
mod quadratic;

use crate::config::FunctionConfig;
use anyhow::Result;
pub use conjugates::{
    build_conjugate_function, L1ConjugateFunction, L2ConjugateFunction,
    QuadraticConjugateFunction,
};
pub use elastic_net::ElasticNetFunction;
pub use hinge_loss::HingeLossFunction;
pub use huber::HuberFunction;
pub use indicator_box::IndicatorBoxFunction;
pub use indicator_simplex::IndicatorSimplexFunction;
pub use l1::L1Function;
pub use l2::L2Function;
pub use logistic_loss::LogisticLossFunction;
pub use max_affine::MaxAffineFunction;
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

    /// Devuelve alpha si la función incluye el término alpha ||x||₁.
    /// El backend OSQP usa variables auxiliares para linealizar el valor absoluto.
    fn l1_alpha(&self) -> Option<f64> {
        None
    }

    /// Devuelve cotas inferiores/superiores si la función es un indicador de caja.
    fn box_bounds(&self, _dimension: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        None
    }

    /// Indica si la función actúa como indicador del simplex probabilístico.
    fn simplex_constraint(&self) -> bool {
        false
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
        FunctionConfig::IndicatorBox { lower, upper } => Ok(Box::new(IndicatorBoxFunction::new(
            lower.clone(),
            upper.clone(),
        )?)),
        FunctionConfig::IndicatorSimplex { tolerance } => {
            Ok(Box::new(IndicatorSimplexFunction::new(*tolerance)?))
        }
        FunctionConfig::ElasticNet {
            l1_alpha,
            l2_alpha,
        } => Ok(Box::new(ElasticNetFunction::new(*l1_alpha, *l2_alpha)?)),
        FunctionConfig::Huber { delta, weight } => {
            Ok(Box::new(HuberFunction::new(*delta, *weight)?))
        }
        FunctionConfig::HingeLoss {
            samples,
            labels,
            weight,
        } => Ok(Box::new(HingeLossFunction::new(
            samples.clone(),
            labels.clone(),
            *weight,
        )?)),
        FunctionConfig::LogisticLoss {
            samples,
            labels,
            l2_alpha,
            weight,
        } => Ok(Box::new(LogisticLossFunction::new(
            samples.clone(),
            labels.clone(),
            *l2_alpha,
            *weight,
        )?)),
        FunctionConfig::MaxAffine { pieces } => {
            Ok(Box::new(MaxAffineFunction::new(pieces.clone())?))
        }
    }
}
