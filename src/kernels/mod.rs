mod squared_norm;

use crate::config::KernelConfig;
use anyhow::Result;
pub use squared_norm::SquaredNormKernel;

/// Interfaz común para kernels convexos.
pub trait KernelFunction {
    fn name(&self) -> &'static str;
    fn value(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
}

pub fn build_kernel(config: &KernelConfig) -> Result<Box<dyn KernelFunction>> {
    match config {
        KernelConfig::SquaredNorm => Ok(Box::new(SquaredNormKernel)),
    }
}
