use crate::kernels::KernelFunction;
use crate::math::norm2_squared;

/// Kernel cuadrático: g(x) = 1/2 ||x||².
pub struct SquaredNormKernel;

impl KernelFunction for SquaredNormKernel {
    fn name(&self) -> &'static str {
        "squared-norm"
    }

    fn value(&self, x: &[f64]) -> f64 {
        0.5 * norm2_squared(x)
    }

    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        // El gradiente de 1/2 ||x||² es x.
        x.to_vec()
    }
}
