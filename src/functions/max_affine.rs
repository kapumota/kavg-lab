use crate::config::AffinePieceConfig;
use crate::functions::ConvexFunction;
use crate::math::dot;
use anyhow::Result;

/// Máximo de funciones afines: f(x)=max_i <a_i,x>+b_i.
pub struct MaxAffineFunction {
    pieces: Vec<AffinePieceConfig>,
}

impl MaxAffineFunction {
    pub fn new(pieces: Vec<AffinePieceConfig>) -> Result<Self> {
        anyhow::ensure!(!pieces.is_empty(), "pieces no puede estar vacío.");
        let dimension = pieces[0].slope.len();
        anyhow::ensure!(
            pieces.iter().all(|piece| piece.slope.len() == dimension),
            "todos los slopes deben tener la misma dimensión."
        );
        Ok(Self { pieces })
    }

    fn active_piece(&self, x: &[f64]) -> &AffinePieceConfig {
        self.pieces
            .iter()
            .max_by(|a, b| {
                let va = dot(&a.slope, x) + a.intercept;
                let vb = dot(&b.slope, x) + b.intercept;
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("MaxAffineFunction requiere al menos una pieza")
    }
}

impl ConvexFunction for MaxAffineFunction {
    fn name(&self) -> &'static str {
        "max-affine"
    }

    fn value(&self, x: &[f64]) -> f64 {
        let piece = self.active_piece(x);
        dot(&piece.slope, x) + piece.intercept
    }

    fn subgradient(&self, x: &[f64]) -> Vec<f64> {
        self.active_piece(x).slope.clone()
    }
}
