mod bregman_quadratic;
mod entropy_kl;
mod huber;
mod mahalanobis;
mod squared_norm;
mod weighted_squared_norm;

use crate::config::KernelConfig;
use anyhow::Result;
pub use bregman_quadratic::BregmanQuadraticKernel;
pub use entropy_kl::EntropyKlKernel;
pub use huber::HuberKernel;
pub use mahalanobis::MahalanobisKernel;
pub use squared_norm::SquaredNormKernel;
pub use weighted_squared_norm::WeightedSquaredNormKernel;

/// Forma cuadrática densa de un kernel: g(x)=1/2 xᵀHx + qᵀx + c.
/// OSQP la usa para kernels cuadráticos generales, incluido Mahalanobis.
#[derive(Debug, Clone)]
pub struct KernelQuadraticForm {
    pub hessian: Vec<Vec<f64>>,
    pub linear: Vec<f64>,
    pub constant: f64,
}

/// Interfaz común para kernels convexos.
pub trait KernelFunction {
    fn name(&self) -> &'static str;
    fn value(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;

    fn quadratic_form(&self, _dimension: usize) -> Option<KernelQuadraticForm> {
        None
    }
}

pub fn build_kernel(config: &KernelConfig) -> Result<Box<dyn KernelFunction>> {
    match config {
        KernelConfig::SquaredNorm => Ok(Box::new(SquaredNormKernel)),
        KernelConfig::WeightedSquaredNorm { weights } => {
            Ok(Box::new(WeightedSquaredNormKernel::new(weights.clone())?))
        }
        KernelConfig::Mahalanobis { matrix } => {
            Ok(Box::new(MahalanobisKernel::new(matrix.clone())?))
        }
        KernelConfig::Huber { delta, weight } => Ok(Box::new(HuberKernel::new(*delta, *weight)?)),
        KernelConfig::EntropyKl { reference, epsilon } => {
            Ok(Box::new(EntropyKlKernel::new(reference.clone(), *epsilon)?))
        }
        KernelConfig::BregmanQuadratic { matrix, center } => Ok(Box::new(
            BregmanQuadraticKernel::new(matrix.clone(), center.clone())?,
        )),
    }
}
