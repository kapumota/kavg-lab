use crate::functions::{ConvexFunction, QuadraticForm};
use crate::math::{norm2, sub};
use crate::optimization::kernel_average::{compute_y2, KernelAverageInput};
use anyhow::{Context, Result};
use osqp::{CscMatrix, Problem, Settings};

const INF: f64 = 1.0e20;

/// Resultado mínimo que el backend OSQP entrega al solver principal.
pub struct OsqpAverageSolution {
    pub y1: Vec<f64>,
    pub iterations: usize,
    pub metric: f64,
}

/// Resuelve el problema del kernel average como QP usando OSQP.
///
/// Variables:
/// z = [y1, y2, t1?, t2?]
/// donde t1/t2 aparecen si f1/f2 incluyen términos L1.
///
/// Restricción central:
/// lambda1 y1 + lambda2 y2 = x.
///
/// En Fase 2 se agregan restricciones de caja, simplex y kernels cuadráticos generales.
pub fn solve_with_osqp(
    input: KernelAverageInput<'_>,
    lambda1: f64,
    lambda2: f64,
) -> Result<OsqpAverageSolution> {
    let n = input.x.len();
    let f1_kind = SupportedFunction::from_function(input.f1, n)
        .context("OSQP no soporta f1 en esta configuración.")?;
    let f2_kind = SupportedFunction::from_function(input.f2, n)
        .context("OSQP no soporta f2 en esta configuración.")?;
    let kernel_form = input
        .kernel
        .quadratic_form(n)
        .with_context(|| format!("OSQP requiere un kernel cuadrático; kernel actual: {}", input.kernel.name()))?;

    let t1_offset = if f1_kind.l1_alpha.is_some() {
        Some(2 * n)
    } else {
        None
    };
    let t2_offset = if f2_kind.l1_alpha.is_some() {
        Some(2 * n + if t1_offset.is_some() { n } else { 0 })
    } else {
        None
    };
    let var_count =
        2 * n + if t1_offset.is_some() { n } else { 0 } + if t2_offset.is_some() { n } else { 0 };

    let mut p_dense = vec![vec![0.0; var_count]; var_count];
    let mut q = vec![0.0; var_count];

    add_supported_function_objective(&mut p_dense, &mut q, &f1_kind, lambda1, 0, n);
    add_supported_function_objective(&mut p_dense, &mut q, &f2_kind, lambda2, n, n);

    if let Some(alpha) = f1_kind.l1_alpha {
        let offset = t1_offset.expect("offset L1 f1 no inicializado");
        for i in 0..n {
            q[offset + i] += lambda1 * alpha;
        }
    }
    if let Some(alpha) = f2_kind.l1_alpha {
        let offset = t2_offset.expect("offset L1 f2 no inicializado");
        for i in 0..n {
            q[offset + i] += lambda2 * alpha;
        }
    }

    // Kernel cuadrático general:
    // g(d)=1/2 dᵀHd + qkᵀd + c, d=y1-y2.
    // El factor cte no afecta la solución y se recalcula fuera de OSQP.
    let penalty_factor = input.average_kind.penalty_factor();
    let c = penalty_factor * lambda1 * lambda2;
    if c != 0.0 {
        for i in 0..n {
            q[i] += c * kernel_form.linear[i];
            q[n + i] -= c * kernel_form.linear[i];
            for j in 0..n {
                let h = c * kernel_form.hessian[i][j];
                p_dense[i][j] += h;
                p_dense[n + i][n + j] += h;
                p_dense[i][n + j] -= h;
                p_dense[n + i][j] -= h;
            }
        }
        let _ = kernel_form.constant;
    }

    let mut a_rows: Vec<Vec<f64>> = Vec::new();
    let mut lower: Vec<f64> = Vec::new();
    let mut upper: Vec<f64> = Vec::new();

    // Igualdad: lambda1 y1 + lambda2 y2 = x.
    for i in 0..n {
        let mut row = vec![0.0; var_count];
        row[i] = lambda1;
        row[n + i] = lambda2;
        a_rows.push(row);
        lower.push(input.x[i]);
        upper.push(input.x[i]);
    }

    if let Some(offset) = t1_offset {
        add_l1_constraints(&mut a_rows, &mut lower, &mut upper, var_count, 0, offset, n);
    }
    if let Some(offset) = t2_offset {
        add_l1_constraints(&mut a_rows, &mut lower, &mut upper, var_count, n, offset, n);
    }

    add_function_constraints(&mut a_rows, &mut lower, &mut upper, var_count, 0, n, &f1_kind);
    add_function_constraints(&mut a_rows, &mut lower, &mut upper, var_count, n, n, &f2_kind);

    let p_iter = p_dense.iter().flat_map(|row| row.iter().copied());
    let a_iter = a_rows.iter().flat_map(|row| row.iter().copied());
    let p = CscMatrix::from_row_iter_dense(var_count, var_count, p_iter).into_upper_tri();
    let a = CscMatrix::from_row_iter_dense(a_rows.len(), var_count, a_iter);

    let settings = Settings::default().verbose(false);
    let mut problem = Problem::new(p, &q, a, &lower, &upper, &settings)
        .map_err(|err| anyhow::anyhow!("No se pudo inicializar OSQP: {err}"))?;
    let result = problem.solve();
    let solution = result
        .x()
        .ok_or_else(|| anyhow::anyhow!("OSQP no encontró una solución primal utilizable."))?;

    let y1 = solution[0..n].to_vec();
    let y2 = solution[n..2 * n].to_vec();
    let y2_from_constraint = compute_y2(input.x, &y1, lambda1, lambda2);
    let residual = norm2(&sub(&y2, &y2_from_constraint));

    // OSQP no expone siempre el número de iteraciones en la API simple.
    // Guardamos max_iterations como cota informativa y usamos el residuo de la restricción como métrica.
    Ok(OsqpAverageSolution {
        y1,
        iterations: input.solver.max_iterations,
        metric: residual,
    })
}

struct SupportedFunction {
    quadratic: Option<QuadraticForm>,
    l1_alpha: Option<f64>,
    box_bounds: Option<(Vec<f64>, Vec<f64>)>,
    simplex: bool,
}

impl SupportedFunction {
    fn from_function(function: &dyn ConvexFunction, dimension: usize) -> Result<Self> {
        let quadratic = function.quadratic_form(dimension);
        let l1_alpha = function.l1_alpha();
        let box_bounds = function.box_bounds(dimension);
        let simplex = function.simplex_constraint();
        anyhow::ensure!(
            quadratic.is_some() || l1_alpha.is_some() || box_bounds.is_some() || simplex,
            "OSQP soporta quadratic, l2, l1, elastic-net, indicator-box, indicator-simplex y conjugados compatibles."
        );
        Ok(Self {
            quadratic,
            l1_alpha,
            box_bounds,
            simplex,
        })
    }
}

fn add_supported_function_objective(
    p_dense: &mut [Vec<f64>],
    q: &mut [f64],
    function: &SupportedFunction,
    weight: f64,
    offset: usize,
    dimension: usize,
) {
    if let Some(form) = &function.quadratic {
        for i in 0..dimension {
            q[offset + i] += weight * form.linear[i];
            for j in 0..dimension {
                p_dense[offset + i][offset + j] += weight * form.hessian[i][j];
            }
        }
        // La constante no afecta la solución; el valor final se recalcula fuera de OSQP.
        let _ = form.constant;
    }
}

fn add_l1_constraints(
    a_rows: &mut Vec<Vec<f64>>,
    lower: &mut Vec<f64>,
    upper: &mut Vec<f64>,
    var_count: usize,
    y_offset: usize,
    t_offset: usize,
    dimension: usize,
) {
    for i in 0..dimension {
        // y_i - t_i <= 0
        let mut row = vec![0.0; var_count];
        row[y_offset + i] = 1.0;
        row[t_offset + i] = -1.0;
        a_rows.push(row);
        lower.push(-INF);
        upper.push(0.0);

        // -y_i - t_i <= 0
        let mut row = vec![0.0; var_count];
        row[y_offset + i] = -1.0;
        row[t_offset + i] = -1.0;
        a_rows.push(row);
        lower.push(-INF);
        upper.push(0.0);
    }
}

fn add_function_constraints(
    a_rows: &mut Vec<Vec<f64>>,
    lower: &mut Vec<f64>,
    upper: &mut Vec<f64>,
    var_count: usize,
    y_offset: usize,
    dimension: usize,
    function: &SupportedFunction,
) {
    if let Some((lo, hi)) = &function.box_bounds {
        for i in 0..dimension {
            let mut row = vec![0.0; var_count];
            row[y_offset + i] = 1.0;
            a_rows.push(row);
            lower.push(lo[i]);
            upper.push(hi[i]);
        }
    }

    if function.simplex {
        for i in 0..dimension {
            let mut row = vec![0.0; var_count];
            row[y_offset + i] = 1.0;
            a_rows.push(row);
            lower.push(0.0);
            upper.push(INF);
        }

        let mut row = vec![0.0; var_count];
        for i in 0..dimension {
            row[y_offset + i] = 1.0;
        }
        a_rows.push(row);
        lower.push(1.0);
        upper.push(1.0);
    }
}
