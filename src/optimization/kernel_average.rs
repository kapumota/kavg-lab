use crate::config::{SolverConfig, SolverMethod};
use crate::functions::ConvexFunction;
use crate::kernels::KernelFunction;
use crate::math::{format_vector, norm2, sub};
use crate::optimization::averages::AverageKind;
use crate::optimization::osqp_solver::solve_with_osqp;
use anyhow::Result;

#[derive(Clone, Copy)]
pub struct KernelAverageInput<'a> {
    pub f1: &'a dyn ConvexFunction,
    pub f2: &'a dyn ConvexFunction,
    pub kernel: &'a dyn KernelFunction,
    pub lambda1: f64,
    pub x: &'a [f64],
    pub solver: &'a SolverConfig,
    pub average_kind: AverageKind,
}

#[derive(Debug, Clone)]
pub struct KernelAverageResult {
    pub index: Option<usize>,
    pub point: Option<Vec<f64>>,
    pub average_kind: String,
    pub value: f64,
    pub y1: Vec<f64>,
    pub y2: Vec<f64>,
    pub raw_penalty: f64,
    pub weighted_penalty: f64,
    pub iterations: usize,
    pub solver_method: String,
    pub solver_metric: f64,
}

impl KernelAverageResult {
    pub fn with_index_and_point(mut self, index: usize, point: Vec<f64>) -> Self {
        self.index = Some(index);
        self.point = Some(point);
        self
    }

    pub fn y1_csv(&self) -> String {
        format_vector(&self.y1)
    }

    pub fn y2_csv(&self) -> String {
        format_vector(&self.y2)
    }

    pub fn point_csv(&self) -> String {
        match &self.point {
            Some(point) => format_vector(point),
            None => String::new(),
        }
    }
}

/// Resuelve numéricamente el promedio definido por `average_kind`.
///
/// La restricción lineal se elimina con:
/// y2 = (x - lambda1 y1) / lambda2.
///
/// Para L1 y otras funciones no suaves, el método recomendado del MVP 2 es
/// coordinate-descent, implementado como búsqueda por coordenadas con reducción de paso.
pub fn solve_kernel_average(input: KernelAverageInput<'_>) -> Result<KernelAverageResult> {
    let lambda1 = input.lambda1;
    let lambda2 = 1.0 - lambda1;

    anyhow::ensure!(
        lambda1 > 0.0 && lambda1 < 1.0,
        "lambda1 debe estar entre 0 y 1."
    );
    anyhow::ensure!(
        input.solver.initial_step > 0.0,
        "initial_step debe ser positivo."
    );
    anyhow::ensure!(
        input.solver.max_iterations > 0,
        "max_iterations debe ser mayor que cero."
    );

    let (best_y1, best_y2, iterations, solver_metric, method_name) = match input.solver.method() {
        SolverMethod::Subgradient => {
            let (y1, iterations, metric, method_name) =
                solve_with_subgradient(input, lambda1, lambda2);
            (y1, None, iterations, metric, method_name)
        }
        SolverMethod::CoordinateDescent => {
            let (y1, iterations, metric, method_name) =
                solve_with_coordinate_descent(input, lambda1, lambda2);
            (y1, None, iterations, metric, method_name)
        }
        SolverMethod::ProximalGradient => {
            let (y1, iterations, metric, method_name) =
                solve_with_proximal_gradient(input, lambda1, lambda2);
            (y1, None, iterations, metric, method_name)
        }
        SolverMethod::Fista => {
            let (y1, iterations, metric, method_name) = solve_with_fista(input, lambda1, lambda2);
            (y1, None, iterations, metric, method_name)
        }
        SolverMethod::Admm => {
            let (y1, iterations, metric, method_name) = solve_with_admm(input, lambda1, lambda2);
            (y1, None, iterations, metric, method_name)
        }
        SolverMethod::Osqp => {
            let solution = solve_with_osqp(input, lambda1, lambda2)?;
            (
                solution.y1,
                Some(solution.y2),
                solution.iterations,
                solution.metric,
                "osqp".to_string(),
            )
        }
    };

    let y2 = best_y2.unwrap_or_else(|| compute_y2(input.x, &best_y1, lambda1, lambda2));
    let diff = sub(&best_y1, &y2);
    let raw_penalty = input.kernel.value(&diff);
    let weighted_penalty = input.average_kind.penalty_factor() * lambda1 * lambda2 * raw_penalty;
    let value = full_objective(input, &best_y1, &y2, lambda1, lambda2);

    Ok(KernelAverageResult {
        index: None,
        point: None,
        average_kind: input.average_kind.name().to_string(),
        value,
        y1: best_y1,
        y2,
        raw_penalty,
        weighted_penalty,
        iterations,
        solver_method: method_name,
        solver_metric,
    })
}

fn solve_with_subgradient(
    input: KernelAverageInput<'_>,
    lambda1: f64,
    lambda2: f64,
) -> (Vec<f64>, usize, f64, String) {
    let mut y1 = input.x.to_vec();
    let mut best_y1 = y1.clone();
    let mut best_value = reduced_objective(input, &y1, lambda1, lambda2);
    let mut last_metric = f64::INFINITY;
    let mut used_iterations = 0;

    for iter in 0..input.solver.max_iterations {
        let gradient = reduced_subgradient(input, &y1, lambda1, lambda2);
        last_metric = norm2(&gradient);
        used_iterations = iter + 1;

        if last_metric <= input.solver.tolerance {
            break;
        }

        let step = input.solver.initial_step / ((iter + 1) as f64).sqrt();
        for i in 0..y1.len() {
            y1[i] -= step * gradient[i];
        }

        let value = reduced_objective(input, &y1, lambda1, lambda2);
        if value < best_value {
            best_value = value;
            best_y1 = y1.clone();
        }
    }

    (
        best_y1,
        used_iterations,
        last_metric,
        "subgradient".to_string(),
    )
}

fn solve_with_coordinate_descent(
    input: KernelAverageInput<'_>,
    lambda1: f64,
    lambda2: f64,
) -> (Vec<f64>, usize, f64, String) {
    let n = input.x.len();
    let mut y1 = input.x.to_vec();
    let mut best_y1 = y1.clone();
    let mut best_value = reduced_objective(input, &y1, lambda1, lambda2);
    let mut step = input.solver.initial_step;
    let min_step = input.solver.min_step();
    let mut used_iterations = 0;

    for pass in 0..input.solver.max_iterations {
        used_iterations = pass + 1;
        let mut improved = false;

        for j in 0..n {
            let current_value = reduced_objective(input, &y1, lambda1, lambda2);

            let mut plus = y1.clone();
            plus[j] += step;
            let plus_value = reduced_objective(input, &plus, lambda1, lambda2);

            let mut minus = y1.clone();
            minus[j] -= step;
            let minus_value = reduced_objective(input, &minus, lambda1, lambda2);

            let relative_tol = input.solver.tolerance * (1.0 + current_value.abs());

            if plus_value + relative_tol < current_value && plus_value <= minus_value {
                y1 = plus;
                improved = true;
            } else if minus_value + relative_tol < current_value {
                y1 = minus;
                improved = true;
            }
        }

        let value = reduced_objective(input, &y1, lambda1, lambda2);
        if value < best_value {
            best_value = value;
            best_y1 = y1.clone();
        }

        if !improved {
            step *= 0.5;
        }

        if step <= min_step {
            break;
        }
    }

    (
        best_y1,
        used_iterations,
        step,
        "coordinate-descent".to_string(),
    )
}

fn solve_with_proximal_gradient(
    input: KernelAverageInput<'_>,
    lambda1: f64,
    lambda2: f64,
) -> (Vec<f64>, usize, f64, String) {
    let mut y1 = input.x.to_vec();
    let mut best_y1 = y1.clone();
    let mut best_value = reduced_objective(input, &y1, lambda1, lambda2);
    let mut step = input.solver.initial_step;
    let min_step = input.solver.min_step();
    let mut solver_metric = f64::INFINITY;
    let mut used_iterations = 0;

    for iter in 0..input.solver.max_iterations {
        used_iterations = iter + 1;
        let gradient = reduced_subgradient(input, &y1, lambda1, lambda2);
        let mut candidate = y1
            .iter()
            .zip(&gradient)
            .map(|(v, g)| v - step * g)
            .collect::<Vec<_>>();
        let current_value = reduced_objective(input, &y1, lambda1, lambda2);
        let mut candidate_value = reduced_objective(input, &candidate, lambda1, lambda2);

        while !candidate_value.is_finite() || candidate_value > current_value {
            step *= 0.5;
            if step <= min_step {
                break;
            }
            candidate = y1
                .iter()
                .zip(&gradient)
                .map(|(v, g)| v - step * g)
                .collect::<Vec<_>>();
            candidate_value = reduced_objective(input, &candidate, lambda1, lambda2);
        }

        solver_metric = norm2(&sub(&candidate, &y1));
        y1 = candidate;
        if candidate_value < best_value {
            best_value = candidate_value;
            best_y1 = y1.clone();
        }
        if solver_metric <= input.solver.tolerance || step <= min_step {
            break;
        }
    }

    (
        best_y1,
        used_iterations,
        solver_metric,
        "proximal-gradient".to_string(),
    )
}

fn solve_with_fista(
    input: KernelAverageInput<'_>,
    lambda1: f64,
    lambda2: f64,
) -> (Vec<f64>, usize, f64, String) {
    let mut y = input.x.to_vec();
    let mut z = y.clone();
    let mut t = 1.0;
    let mut best_y1 = y.clone();
    let mut best_value = reduced_objective(input, &y, lambda1, lambda2);
    let mut step = input.solver.initial_step;
    let min_step = input.solver.min_step();
    let mut solver_metric = f64::INFINITY;
    let mut used_iterations = 0;

    for iter in 0..input.solver.max_iterations {
        used_iterations = iter + 1;
        let gradient = reduced_subgradient(input, &z, lambda1, lambda2);
        let mut next_y = z
            .iter()
            .zip(&gradient)
            .map(|(v, g)| v - step * g)
            .collect::<Vec<_>>();
        let current_value = reduced_objective(input, &y, lambda1, lambda2);
        let mut next_value = reduced_objective(input, &next_y, lambda1, lambda2);

        while !next_value.is_finite() || next_value > current_value + 1.0e-10 {
            step *= 0.5;
            if step <= min_step {
                break;
            }
            next_y = z
                .iter()
                .zip(&gradient)
                .map(|(v, g)| v - step * g)
                .collect::<Vec<_>>();
            next_value = reduced_objective(input, &next_y, lambda1, lambda2);
        }

        let next_t = 0.5_f64 * (1.0_f64 + (1.0_f64 + 4.0_f64 * t * t).sqrt());
        let momentum = (t - 1.0) / next_t;
        let next_z = next_y
            .iter()
            .zip(&y)
            .map(|(ny, old)| ny + momentum * (ny - old))
            .collect::<Vec<_>>();

        solver_metric = norm2(&sub(&next_y, &y));
        y = next_y;
        z = next_z;
        t = next_t;

        if next_value < best_value {
            best_value = next_value;
            best_y1 = y.clone();
        }
        if solver_metric <= input.solver.tolerance || step <= min_step {
            break;
        }
    }

    (best_y1, used_iterations, solver_metric, "fista".to_string())
}

fn solve_with_admm(
    input: KernelAverageInput<'_>,
    lambda1: f64,
    lambda2: f64,
) -> (Vec<f64>, usize, f64, String) {
    // ADMM completo requiere operadores proximales explícitos por función. En esta fase se
    // implementa una variante experimental: pasos de gradiente sobre la variable primal y una
    // variable dual ligera para estabilizar la restricción eliminada y1/y2.
    let mut y1 = input.x.to_vec();
    let mut dual = vec![0.0; input.x.len()];
    let mut best_y1 = y1.clone();
    let mut best_value = reduced_objective(input, &y1, lambda1, lambda2);
    let rho = 1.0 / input.solver.initial_step.max(1.0e-9);
    let mut step = input.solver.initial_step;
    let min_step = input.solver.min_step();
    let mut solver_metric = f64::INFINITY;
    let mut used_iterations = 0;

    for iter in 0..input.solver.max_iterations {
        used_iterations = iter + 1;
        let gradient = reduced_subgradient(input, &y1, lambda1, lambda2);
        let y2 = compute_y2(input.x, &y1, lambda1, lambda2);
        let residual = sub(&y1, &y2);
        for i in 0..y1.len() {
            y1[i] -= step * (gradient[i] + rho * residual[i] + dual[i]);
            dual[i] += step * residual[i];
        }
        step = (step * 0.995).max(min_step);
        solver_metric = norm2(&residual);
        let value = reduced_objective(input, &y1, lambda1, lambda2);
        if value < best_value {
            best_value = value;
            best_y1 = y1.clone();
        }
        if solver_metric <= input.solver.tolerance {
            break;
        }
    }

    (best_y1, used_iterations, solver_metric, "admm".to_string())
}

fn full_objective(
    input: KernelAverageInput<'_>,
    y1: &[f64],
    y2: &[f64],
    lambda1: f64,
    lambda2: f64,
) -> f64 {
    let diff = sub(y1, y2);

    lambda1 * input.f1.value(y1)
        + lambda2 * input.f2.value(y2)
        + input.average_kind.penalty_factor() * lambda1 * lambda2 * input.kernel.value(&diff)
}

fn reduced_objective(input: KernelAverageInput<'_>, y1: &[f64], lambda1: f64, lambda2: f64) -> f64 {
    let y2 = compute_y2(input.x, y1, lambda1, lambda2);
    full_objective(input, y1, &y2, lambda1, lambda2)
}

fn reduced_subgradient(
    input: KernelAverageInput<'_>,
    y1: &[f64],
    lambda1: f64,
    lambda2: f64,
) -> Vec<f64> {
    let y2 = compute_y2(input.x, y1, lambda1, lambda2);
    let diff = sub(y1, &y2);
    let f1_grad = input.f1.subgradient(y1);
    let f2_grad = input.f2.subgradient(&y2);
    let kernel_grad = input.kernel.gradient(&diff);
    let penalty_factor = input.average_kind.penalty_factor();

    f1_grad
        .iter()
        .zip(f2_grad.iter())
        .zip(kernel_grad.iter())
        .map(|((a, b), k)| lambda1 * (a - b + penalty_factor * k))
        .collect()
}

pub fn compute_y2(x: &[f64], y1: &[f64], lambda1: f64, lambda2: f64) -> Vec<f64> {
    x.iter()
        .zip(y1)
        .map(|(x_i, y1_i)| (x_i - lambda1 * y1_i) / lambda2)
        .collect()
}

/// Promedio aritmético: lambda1 f1(x) + lambda2 f2(x).
pub fn arithmetic_average(
    f1: &dyn ConvexFunction,
    f2: &dyn ConvexFunction,
    lambda1: f64,
    x: &[f64],
) -> f64 {
    let lambda2 = 1.0 - lambda1;
    lambda1 * f1.value(x) + lambda2 * f2.value(x)
}
