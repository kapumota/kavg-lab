use crate::config::SolverConfig;
use crate::functions::ConvexFunction;
use crate::kernels::KernelFunction;
use crate::math::{dot, format_vector, norm2, sub};
use crate::optimization::averages::AverageKind;
use crate::optimization::kernel_average::{solve_kernel_average, KernelAverageInput};
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct FenchelCheckResult {
    pub index: usize,
    pub dual_point: Vec<f64>,
    pub left_approx: f64,
    pub right_value: f64,
    pub absolute_error: f64,
    pub relative_error: f64,
    pub passed: bool,
    pub primal_argmax: Vec<f64>,
    pub right_y1: Vec<f64>,
    pub right_y2: Vec<f64>,
    pub outer_iterations: usize,
    pub outer_metric: f64,
}

impl FenchelCheckResult {
    pub fn dual_point_csv(&self) -> String {
        format_vector(&self.dual_point)
    }

    pub fn primal_argmax_csv(&self) -> String {
        format_vector(&self.primal_argmax)
    }

    pub fn right_y1_csv(&self) -> String {
        format_vector(&self.right_y1)
    }

    pub fn right_y2_csv(&self) -> String {
        format_vector(&self.right_y2)
    }
}

/// Entrada agrupada para verificar la identidad de Fenchel sin exponer una firma larga.
pub struct FenchelIdentityInput<'a> {
    pub index: usize,
    pub dual_point: &'a [f64],
    pub f1: &'a dyn ConvexFunction,
    pub f2: &'a dyn ConvexFunction,
    pub f1_star: &'a dyn ConvexFunction,
    pub f2_star: &'a dyn ConvexFunction,
    pub kernel: &'a dyn KernelFunction,
    pub lambda1: f64,
    pub solver: &'a SolverConfig,
}

/// Verifica numéricamente la identidad:
///
/// (P(f1,f2,g))*(s) ≈ P(f1*,f2*,g*)(s).
///
/// El lado izquierdo se aproxima con una maximización numérica:
/// sup_x <s,x> - P(f1,f2,g)(x).
/// El lado derecho usa conjugados analíticos de f1 y f2.
pub fn verify_fenchel_identity(input: FenchelIdentityInput<'_>) -> Result<FenchelCheckResult> {
    let (left_approx, primal_argmax, outer_iterations, outer_metric) = approximate_conjugate(
        input.dual_point,
        input.f1,
        input.f2,
        input.kernel,
        input.lambda1,
        input.solver,
    )?;

    // Para g(x)=1/2||x||² se tiene g*=g y g^{*∨}=g.
    let right = solve_kernel_average(KernelAverageInput {
        f1: input.f1_star,
        f2: input.f2_star,
        kernel: input.kernel,
        lambda1: input.lambda1,
        x: input.dual_point,
        solver: input.solver,
        average_kind: AverageKind::Kernel,
    })?;

    let right_value = right.value;
    let absolute_error = (left_approx - right_value).abs();
    let relative_error = absolute_error / (1.0 + right_value.abs());
    let tolerance = input.solver.tolerance.max(1.0e-8);
    let passed = absolute_error <= 20.0 * tolerance || relative_error <= 20.0 * tolerance;

    Ok(FenchelCheckResult {
        index: input.index,
        dual_point: input.dual_point.to_vec(),
        left_approx,
        right_value,
        absolute_error,
        relative_error,
        passed,
        primal_argmax,
        right_y1: right.y1,
        right_y2: right.y2,
        outer_iterations,
        outer_metric,
    })
}

fn approximate_conjugate(
    dual_point: &[f64],
    f1: &dyn ConvexFunction,
    f2: &dyn ConvexFunction,
    kernel: &dyn KernelFunction,
    lambda1: f64,
    solver: &SolverConfig,
) -> Result<(f64, Vec<f64>, usize, f64)> {
    let n = dual_point.len();
    let mut x = dual_point.to_vec();
    let mut best_x = x.clone();
    let mut best_value = conjugate_objective(dual_point, &x, f1, f2, kernel, lambda1, solver)?;
    let mut step = solver.initial_step.max(1.0);
    let min_step = solver.min_step();
    let mut used_iterations = 0;

    for iter in 0..solver.max_iterations {
        used_iterations = iter + 1;
        let mut improved = false;

        for j in 0..n {
            let current_value =
                conjugate_objective(dual_point, &x, f1, f2, kernel, lambda1, solver)?;

            let mut plus = x.clone();
            plus[j] += step;
            let plus_value =
                conjugate_objective(dual_point, &plus, f1, f2, kernel, lambda1, solver)?;

            let mut minus = x.clone();
            minus[j] -= step;
            let minus_value =
                conjugate_objective(dual_point, &minus, f1, f2, kernel, lambda1, solver)?;

            let relative_tol = solver.tolerance * (1.0 + current_value.abs());

            if plus_value > current_value + relative_tol && plus_value >= minus_value {
                x = plus;
                improved = true;
            } else if minus_value > current_value + relative_tol {
                x = minus;
                improved = true;
            }
        }

        let value = conjugate_objective(dual_point, &x, f1, f2, kernel, lambda1, solver)?;
        if value > best_value {
            best_value = value;
            best_x = x.clone();
        }

        if !improved {
            step *= 0.5;
        }

        if step <= min_step {
            break;
        }
    }

    // Métrica simple: norma del movimiento entre el mejor punto encontrado y el punto dual.
    let displacement = sub(&best_x, dual_point);
    Ok((best_value, best_x, used_iterations, norm2(&displacement)))
}

fn conjugate_objective(
    dual_point: &[f64],
    x: &[f64],
    f1: &dyn ConvexFunction,
    f2: &dyn ConvexFunction,
    kernel: &dyn KernelFunction,
    lambda1: f64,
    solver: &SolverConfig,
) -> Result<f64> {
    let primal = solve_kernel_average(KernelAverageInput {
        f1,
        f2,
        kernel,
        lambda1,
        x,
        solver,
        average_kind: AverageKind::Kernel,
    })?;

    Ok(dot(dual_point, x) - primal.value)
}
