use crate::config::{SolverConfig, SolverMethod};
use crate::functions::ConvexFunction;
use crate::kernels::KernelFunction;
use crate::math::format_vector;
use crate::optimization::averages::AverageKind;
use crate::optimization::kernel_average::{solve_kernel_average, KernelAverageInput};
use crate::parallel::{self, ExecutionMode};

#[derive(Debug, Clone)]
pub struct SolverComparisonRow {
    pub solver_method: String,
    pub index: usize,
    pub point: Vec<f64>,
    pub status: String,
    pub value: Option<f64>,
    pub iterations: Option<usize>,
    pub solver_metric: Option<f64>,
    pub raw_penalty: Option<f64>,
    pub weighted_penalty: Option<f64>,
    pub y1: Vec<f64>,
    pub y2: Vec<f64>,
    pub error: Option<String>,
}

impl SolverComparisonRow {
    pub fn point_csv(&self) -> String {
        format_vector(&self.point)
    }

    pub fn y1_csv(&self) -> String {
        if self.y1.is_empty() {
            String::new()
        } else {
            format_vector(&self.y1)
        }
    }

    pub fn y2_csv(&self) -> String {
        if self.y2.is_empty() {
            String::new()
        } else {
            format_vector(&self.y2)
        }
    }
}

pub fn compare_solvers_for_points(
    points: &[Vec<f64>],
    f1: &dyn ConvexFunction,
    f2: &dyn ConvexFunction,
    kernel: &dyn KernelFunction,
    lambda1: f64,
    base_solver: &SolverConfig,
    methods: &[SolverMethod],
) -> Vec<SolverComparisonRow> {
    compare_solvers_for_points_with_mode(
        points,
        f1,
        f2,
        kernel,
        lambda1,
        base_solver,
        methods,
        ExecutionMode::Sequential,
    )
    .unwrap_or_else(|error| {
        vec![SolverComparisonRow {
            solver_method: "internal".to_string(),
            index: 0,
            point: Vec::new(),
            status: "error".to_string(),
            value: None,
            iterations: None,
            solver_metric: None,
            raw_penalty: None,
            weighted_penalty: None,
            y1: Vec::new(),
            y2: Vec::new(),
            error: Some(error.to_string()),
        }]
    })
}

#[allow(clippy::too_many_arguments)]
pub fn compare_solvers_for_points_with_mode(
    points: &[Vec<f64>],
    f1: &dyn ConvexFunction,
    f2: &dyn ConvexFunction,
    kernel: &dyn KernelFunction,
    lambda1: f64,
    base_solver: &SolverConfig,
    methods: &[SolverMethod],
    mode: ExecutionMode,
) -> anyhow::Result<Vec<SolverComparisonRow>> {
    let tasks: Vec<(SolverMethod, usize, Vec<f64>)> = methods
        .iter()
        .flat_map(|method| {
            points
                .iter()
                .enumerate()
                .map(move |(index, point)| (method.clone(), index, point.clone()))
        })
        .collect();

    parallel::map_indexed(&tasks, mode, |_order, task| {
        let (method, index, point) = task;
        let solver = SolverConfig {
            method: Some(method.clone()),
            initial_step: base_solver.initial_step,
            tolerance: base_solver.tolerance,
            min_step: base_solver.min_step,
            max_iterations: base_solver.max_iterations,
        };

        let result = solve_kernel_average(KernelAverageInput {
            f1,
            f2,
            kernel,
            lambda1,
            x: point,
            solver: &solver,
            average_kind: AverageKind::Kernel,
        });

        Ok(match result {
            Ok(result) => SolverComparisonRow {
                solver_method: method.as_str().to_string(),
                index: *index,
                point: point.clone(),
                status: "ok".to_string(),
                value: Some(result.value),
                iterations: Some(result.iterations),
                solver_metric: Some(result.solver_metric),
                raw_penalty: Some(result.raw_penalty),
                weighted_penalty: Some(result.weighted_penalty),
                y1: result.y1,
                y2: result.y2,
                error: None,
            },
            Err(error) => SolverComparisonRow {
                solver_method: method.as_str().to_string(),
                index: *index,
                point: point.clone(),
                status: "error".to_string(),
                value: None,
                iterations: None,
                solver_metric: None,
                raw_penalty: None,
                weighted_penalty: None,
                y1: Vec::new(),
                y2: Vec::new(),
                error: Some(error.to_string()),
            },
        })
    })
}
