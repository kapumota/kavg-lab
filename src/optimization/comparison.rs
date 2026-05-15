use crate::config::SolverConfig;
use crate::functions::ConvexFunction;
use crate::kernels::KernelFunction;
use crate::optimization::averages::AverageKind;
use crate::optimization::kernel_average::{
    arithmetic_average, solve_kernel_average, KernelAverageInput, KernelAverageResult,
};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub index: usize,
    pub point: Vec<f64>,
    pub arithmetic_value: f64,
    pub epigraphical: KernelAverageResult,
    pub proximal: KernelAverageResult,
}

/// Calcula los tres promedios principales del MVP 2 en un punto.
pub fn compare_averages(
    index: usize,
    point: &[f64],
    f1: &dyn ConvexFunction,
    f2: &dyn ConvexFunction,
    kernel: &dyn KernelFunction,
    lambda1: f64,
    solver: &SolverConfig,
) -> Result<ComparisonResult> {
    let arithmetic_value = arithmetic_average(f1, f2, lambda1, point);

    let epigraphical = solve_kernel_average(KernelAverageInput {
        f1,
        f2,
        kernel,
        lambda1,
        x: point,
        solver,
        average_kind: AverageKind::Epigraphical,
    })?
    .with_index_and_point(index, point.to_vec());

    let proximal = solve_kernel_average(KernelAverageInput {
        f1,
        f2,
        kernel,
        lambda1,
        x: point,
        solver,
        average_kind: AverageKind::Proximal,
    })?
    .with_index_and_point(index, point.to_vec());

    Ok(ComparisonResult {
        index,
        point: point.to_vec(),
        arithmetic_value,
        epigraphical,
        proximal,
    })
}
