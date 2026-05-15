/// Tipos de promedio soportados por el MVP 2.
#[derive(Debug, Clone, Copy)]
pub enum AverageKind {
    /// Promedio epigráfico: no agrega penalización kernel.
    Epigraphical,
    /// Proximal average: usa g(x)=1/2||x||².
    Proximal,
    /// Kernel average general del proyecto; en el MVP 2 coincide con proximal porque solo existe squared-norm.
    Kernel,
}

impl AverageKind {
    pub fn name(&self) -> &'static str {
        match self {
            AverageKind::Epigraphical => "epigraphical",
            AverageKind::Proximal => "proximal",
            AverageKind::Kernel => "kernel",
        }
    }

    /// Factor que multiplica el término lambda1 lambda2 g(y1-y2).
    pub fn penalty_factor(&self) -> f64 {
        match self {
            AverageKind::Epigraphical => 0.0,
            AverageKind::Proximal => 1.0,
            AverageKind::Kernel => 1.0,
        }
    }
}
