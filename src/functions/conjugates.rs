use crate::config::FunctionConfig;
use crate::functions::{ConvexFunction, QuadraticForm};
use crate::math::{dot, norm2_squared, solve_linear_system, sub, transpose};
use anyhow::Result;

/// Conjugado de f(x) = 1/2 ||Ax - b||² cuando A es cuadrada e invertible.
///
/// Si p = A^{-T}s, entonces f*(s) = 1/2 ||p||² + <p,b>.
pub struct QuadraticConjugateFunction {
    matrix: Vec<Vec<f64>>,
    matrix_t: Vec<Vec<f64>>,
    vector: Vec<f64>,
}

impl QuadraticConjugateFunction {
    pub fn new(matrix: Vec<Vec<f64>>, vector: Vec<f64>) -> Result<Self> {
        anyhow::ensure!(!matrix.is_empty(), "La matriz no puede estar vacía.");
        let n = matrix.len();
        anyhow::ensure!(
            matrix.iter().all(|row| row.len() == n),
            "Para el conjugado analítico de quadratic, A debe ser cuadrada."
        );
        anyhow::ensure!(
            vector.len() == n,
            "La dimensión de b debe coincidir con la matriz cuadrada A."
        );

        let matrix_t = transpose(&matrix);
        // Validamos invertibilidad resolviendo un sistema simple. Si falla, reportamos temprano.
        let _ = solve_linear_system(&matrix_t, &vec![0.0; n])?;

        Ok(Self {
            matrix,
            matrix_t,
            vector,
        })
    }

    fn solve_p(&self, s: &[f64]) -> Result<Vec<f64>> {
        solve_linear_system(&self.matrix_t, s)
    }
}

impl ConvexFunction for QuadraticConjugateFunction {
    fn name(&self) -> &'static str {
        "quadratic-conjugate"
    }

    fn value(&self, s: &[f64]) -> f64 {
        let p = self
            .solve_p(s)
            .expect("No se pudo evaluar el conjugado cuadrático: A debe ser invertible.");
        0.5 * norm2_squared(&p) + dot(&p, &self.vector)
    }

    fn subgradient(&self, s: &[f64]) -> Vec<f64> {
        // ∇f*(s) = A^{-1}(A^{-T}s + b).
        let p = self
            .solve_p(s)
            .expect("No se pudo calcular el gradiente del conjugado cuadrático.");
        let rhs: Vec<f64> = p
            .iter()
            .zip(self.vector.iter())
            .map(|(p_i, b_i)| p_i + b_i)
            .collect();
        solve_linear_system(&self.matrix, &rhs).expect("No se pudo resolver A q = A^{-T}s + b.")
    }

    fn quadratic_form(&self, dimension: usize) -> Option<QuadraticForm> {
        if self.matrix.len() != dimension {
            return None;
        }

        // Usamos la identidad ∇f*(s) = Hs + q. Calculamos H por columnas.
        let zero = vec![0.0; dimension];
        let linear = self.subgradient(&zero);
        let mut hessian = vec![vec![0.0; dimension]; dimension];

        for col in 0..dimension {
            let mut e = vec![0.0; dimension];
            e[col] = 1.0;
            let grad_e = self.subgradient(&e);
            let column = sub(&grad_e, &linear);
            for row in 0..dimension {
                hessian[row][col] = column[row];
            }
        }

        Some(QuadraticForm {
            hessian,
            linear,
            constant: self.value(&zero),
        })
    }
}

/// Conjugado de f(x) = (alpha/2)||x||² con alpha > 0.
/// El conjugado es f*(s) = (1/(2alpha))||s||².
pub struct L2ConjugateFunction {
    alpha: f64,
}

impl L2ConjugateFunction {
    pub fn new(alpha: f64) -> Result<Self> {
        anyhow::ensure!(
            alpha > 0.0,
            "Para el conjugado de L2 se requiere alpha > 0."
        );
        Ok(Self { alpha })
    }
}

impl ConvexFunction for L2ConjugateFunction {
    fn name(&self) -> &'static str {
        "l2-conjugate"
    }

    fn value(&self, s: &[f64]) -> f64 {
        0.5 * norm2_squared(s) / self.alpha
    }

    fn subgradient(&self, s: &[f64]) -> Vec<f64> {
        s.iter().map(|v| v / self.alpha).collect()
    }

    fn quadratic_form(&self, dimension: usize) -> Option<QuadraticForm> {
        let mut hessian = vec![vec![0.0; dimension]; dimension];
        for (i, row) in hessian.iter_mut().enumerate().take(dimension) {
            row[i] = 1.0 / self.alpha;
        }
        Some(QuadraticForm {
            hessian,
            linear: vec![0.0; dimension],
            constant: 0.0,
        })
    }
}

/// Conjugado de f(x)=alpha||x||₁.
/// Es el indicador de la bola infinito: f*(s)=0 si ||s||∞ <= alpha, +∞ en otro caso.
pub struct L1ConjugateFunction {
    alpha: f64,
}

impl L1ConjugateFunction {
    pub fn new(alpha: f64) -> Result<Self> {
        anyhow::ensure!(alpha >= 0.0, "alpha debe ser no negativo.");
        Ok(Self { alpha })
    }
}

impl ConvexFunction for L1ConjugateFunction {
    fn name(&self) -> &'static str {
        "l1-conjugate-indicator-linf-ball"
    }

    fn value(&self, s: &[f64]) -> f64 {
        if s.iter().all(|value| value.abs() <= self.alpha + 1.0e-12) {
            0.0
        } else {
            f64::INFINITY
        }
    }

    fn subgradient(&self, _s: &[f64]) -> Vec<f64> {
        // En el interior de la bola infinito elegimos el subgradiente cero.
        // En la frontera, el normal completo es multivaluado; para el solver QP usamos las cotas.
        vec![0.0; _s.len()]
    }

    fn box_bounds(&self, dimension: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        Some((vec![-self.alpha; dimension], vec![self.alpha; dimension]))
    }
}

/// Construye el conjugado analítico de una función soportada a partir de su configuración.
pub fn build_conjugate_function(config: &FunctionConfig) -> Result<Box<dyn ConvexFunction>> {
    match config {
        FunctionConfig::Quadratic { matrix, vector } => Ok(Box::new(
            QuadraticConjugateFunction::new(matrix.clone(), vector.clone())?,
        )),
        FunctionConfig::L2 { alpha } => Ok(Box::new(L2ConjugateFunction::new(*alpha)?)),
        FunctionConfig::L1 { alpha } => Ok(Box::new(L1ConjugateFunction::new(*alpha)?)),
        FunctionConfig::ElasticNet { .. }
        | FunctionConfig::IndicatorBox { .. }
        | FunctionConfig::IndicatorSimplex { .. }
        | FunctionConfig::Huber { .. }
        | FunctionConfig::HingeLoss { .. }
        | FunctionConfig::LogisticLoss { .. }
        | FunctionConfig::MaxAffine { .. } => anyhow::bail!(
            "El conjugado analítico todavía no está implementado para esta función de Fase 2. Use quadratic, l2 o l1 para verify-fenchel."
        ),
    }
}
