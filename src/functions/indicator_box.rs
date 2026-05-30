use crate::functions::ConvexFunction;
use anyhow::Result;

const FEASIBILITY_TOLERANCE: f64 = 1.0e-7;

/// Indicador de una caja: f(x)=0 si lower <= x <= upper, +∞ en otro caso.
pub struct IndicatorBoxFunction {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl IndicatorBoxFunction {
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Result<Self> {
        anyhow::ensure!(
            lower.len() == upper.len(),
            "lower y upper deben tener la misma dimensión."
        );
        anyhow::ensure!(
            lower.iter().zip(&upper).all(|(lo, hi)| lo <= hi),
            "cada lower[i] debe ser menor o igual que upper[i]."
        );
        Ok(Self { lower, upper })
    }
}

impl ConvexFunction for IndicatorBoxFunction {
    fn name(&self) -> &'static str {
        "indicator-box"
    }

    fn value(&self, x: &[f64]) -> f64 {
        if x.len() == self.lower.len()
            && x.iter()
                .zip(&self.lower)
                .zip(&self.upper)
                .all(|((v, lo), hi)| {
                    *v >= *lo - FEASIBILITY_TOLERANCE && *v <= *hi + FEASIBILITY_TOLERANCE
                })
        {
            0.0
        } else {
            f64::INFINITY
        }
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        // Dentro de la caja usamos el subgradiente 0. Fuera, el valor ya es +∞.
        vec![0.0; x.len()]
    }

    fn box_bounds(&self, dimension: usize) -> Option<(Vec<f64>, Vec<f64>)> {
        if self.lower.len() == dimension {
            Some((self.lower.clone(), self.upper.clone()))
        } else {
            None
        }
    }
}
