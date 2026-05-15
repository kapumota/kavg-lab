use anyhow::{Context, Result};
use serde::Deserialize;
use std::{fs, path::Path};

#[derive(Debug, Deserialize)]
pub struct ExperimentConfig {
    pub dimension: usize,
    pub lambda1: f64,
    pub f1: FunctionConfig,
    pub f2: FunctionConfig,
    pub kernel: KernelConfig,
    pub solver: SolverConfig,
    pub points: Vec<Vec<f64>>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum FunctionConfig {
    #[serde(rename = "quadratic")]
    Quadratic {
        matrix: Vec<Vec<f64>>,
        vector: Vec<f64>,
    },
    #[serde(rename = "l1")]
    L1 { alpha: f64 },
    #[serde(rename = "l2")]
    L2 { alpha: f64 },
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum KernelConfig {
    #[serde(rename = "squared-norm")]
    SquaredNorm,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum SolverMethod {
    Subgradient,
    CoordinateDescent,
    Osqp,
}

impl SolverMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            SolverMethod::Subgradient => "subgradient",
            SolverMethod::CoordinateDescent => "coordinate-descent",
            SolverMethod::Osqp => "osqp",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct SolverConfig {
    /// Método numérico. Por defecto se usa coordinate-descent porque maneja bien L1.
    /// Use `osqp` para resolver los casos QP con un backend externo.
    pub method: Option<SolverMethod>,
    /// Tamaño de paso inicial para subgradiente o búsqueda por coordenadas.
    pub initial_step: f64,
    /// Tolerancia usada como criterio de parada.
    pub tolerance: f64,
    /// Paso mínimo para coordinate-descent.
    pub min_step: Option<f64>,
    /// Número máximo de iteraciones o pasadas completas por coordenadas.
    pub max_iterations: usize,
}

impl SolverConfig {
    pub fn method(&self) -> SolverMethod {
        self.method
            .clone()
            .unwrap_or(SolverMethod::CoordinateDescent)
    }

    pub fn min_step(&self) -> f64 {
        self.min_step.unwrap_or(self.tolerance.max(1.0e-12))
    }
}

impl ExperimentConfig {
    pub fn from_yaml_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("No se pudo leer el archivo: {}", path.display()))?;
        let config: ExperimentConfig = serde_yaml::from_str(&text)
            .with_context(|| format!("No se pudo parsear YAML: {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        anyhow::ensure!(self.dimension > 0, "La dimensión debe ser mayor que cero.");
        anyhow::ensure!(
            self.lambda1 > 0.0 && self.lambda1 < 1.0,
            "lambda1 debe estar estrictamente entre 0 y 1."
        );
        anyhow::ensure!(
            !self.points.is_empty(),
            "Debe definirse al menos un punto de evaluación."
        );
        anyhow::ensure!(
            self.solver.initial_step > 0.0,
            "solver.initial_step debe ser positivo."
        );
        anyhow::ensure!(
            self.solver.tolerance > 0.0,
            "solver.tolerance debe ser positivo."
        );
        anyhow::ensure!(
            self.solver.max_iterations > 0,
            "solver.max_iterations debe ser mayor que cero."
        );

        self.validate_function(&self.f1, "f1")?;
        self.validate_function(&self.f2, "f2")?;

        for point in &self.points {
            anyhow::ensure!(
                point.len() == self.dimension,
                "Todos los puntos deben tener dimensión {}.",
                self.dimension
            );
        }
        Ok(())
    }

    fn validate_function(&self, function: &FunctionConfig, label: &str) -> Result<()> {
        match function {
            FunctionConfig::Quadratic { matrix, vector } => {
                anyhow::ensure!(
                    !matrix.is_empty(),
                    "{}: la matriz no puede estar vacía.",
                    label
                );
                anyhow::ensure!(
                    matrix.iter().all(|row| row.len() == self.dimension),
                    "{}: cada fila de la matriz debe tener dimensión {}.",
                    label,
                    self.dimension
                );
                anyhow::ensure!(
                    vector.len() == matrix.len(),
                    "{}: el vector b debe tener tantas entradas como filas tenga A.",
                    label
                );
            }
            FunctionConfig::L1 { alpha } => {
                anyhow::ensure!(*alpha >= 0.0, "{}: alpha debe ser no negativo.", label);
            }
            FunctionConfig::L2 { alpha } => {
                anyhow::ensure!(*alpha >= 0.0, "{}: alpha debe ser no negativo.", label);
            }
        }
        Ok(())
    }
}

/// Método de solución específico para atención.
/// Se mantiene separado del solver convexo porque la demo usa gradiente proyectado.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum AttentionSolverMethod {
    ProjectedGradient,
}

impl AttentionSolverMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            AttentionSolverMethod::ProjectedGradient => "projected-gradient",
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AttentionSolverConfig {
    /// Método numérico de atención. Actualmente solo se implementa projected-gradient.
    pub method: Option<AttentionSolverMethod>,
    /// Tamaño de paso inicial para gradiente proyectado.
    pub initial_step: f64,
    /// Tolerancia usada como criterio de parada.
    pub tolerance: f64,
    /// Paso mínimo usado para evitar pasos degenerados.
    pub min_step: Option<f64>,
    /// Número máximo de iteraciones.
    pub max_iterations: usize,
}

impl AttentionSolverConfig {
    pub fn method(&self) -> AttentionSolverMethod {
        self.method
            .clone()
            .unwrap_or(AttentionSolverMethod::ProjectedGradient)
    }

    pub fn min_step(&self) -> f64 {
        self.min_step.unwrap_or(self.tolerance.max(1.0e-12))
    }

    pub fn validate(&self, label: &str) -> Result<()> {
        anyhow::ensure!(
            self.initial_step > 0.0,
            "{}.initial_step debe ser positivo.",
            label
        );
        anyhow::ensure!(
            self.tolerance > 0.0,
            "{}.tolerance debe ser positivo.",
            label
        );
        anyhow::ensure!(
            self.max_iterations > 0,
            "{}.max_iterations debe ser mayor que cero.",
            label
        );
        if let Some(min_step) = self.min_step {
            anyhow::ensure!(min_step > 0.0, "{}.min_step debe ser positivo.", label);
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum MaskEntry {
    Number(f64),
    Text(String),
}

impl MaskEntry {
    pub fn to_f64(&self) -> Result<f64> {
        match self {
            MaskEntry::Number(value) => Ok(*value),
            MaskEntry::Text(text) => {
                let normalized = text.trim().to_ascii_lowercase();
                match normalized.as_str() {
                    "-inf" | "-.inf" | "-infinity" => Ok(f64::NEG_INFINITY),
                    "inf" | "+inf" | ".inf" | "+.inf" | "infinity" => Ok(f64::INFINITY),
                    _ => text.parse::<f64>().with_context(|| {
                        format!("No se pudo interpretar entrada de máscara: {text}")
                    }),
                }
            }
        }
    }
}

/// Configuración de máscara para demos tipo Transformer/LLM.
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type")]
pub enum AttentionMaskConfig {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "causal")]
    Causal,
    #[serde(rename = "custom")]
    Custom { matrix: Vec<Vec<MaskEntry>> },
}

impl AttentionMaskConfig {
    pub fn validate(&self, rows: usize, cols: usize) -> Result<()> {
        match self {
            AttentionMaskConfig::None => Ok(()),
            AttentionMaskConfig::Causal => {
                anyhow::ensure!(
                    rows == cols,
                    "mask.type=causal requiere queries.len() == keys.len() para una demo autoregresiva tipo LLM."
                );
                Ok(())
            }
            AttentionMaskConfig::Custom { matrix } => {
                anyhow::ensure!(
                    matrix.len() == rows,
                    "mask.matrix debe tener {} filas.",
                    rows
                );
                for row in matrix {
                    anyhow::ensure!(
                        row.len() == cols,
                        "Cada fila de mask.matrix debe tener {} columnas.",
                        cols
                    );
                    for entry in row {
                        entry.to_f64()?;
                    }
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AttentionConfig {
    /// Dimensión de queries y keys.
    pub dimension: usize,
    /// Temperatura usada en los scores q·k/sqrt(d)/temperature.
    pub temperature: f64,
    /// Peso del término kernel gamma/2 ||p - p0||².
    pub kernel_gamma: f64,
    /// Solver usado por gradiente proyectado.
    pub attention_solver: AttentionSolverConfig,
    /// Máscara opcional: none, causal o custom.
    pub mask: Option<AttentionMaskConfig>,
    /// Consultas de atención. En cross-attention multimodal también acepta `text_queries`.
    #[serde(alias = "text_queries")]
    pub queries: Vec<Vec<f64>>,
    /// Keys de atención. En cross-attention multimodal también acepta `image_keys`.
    #[serde(alias = "image_keys")]
    pub keys: Vec<Vec<f64>>,
    /// Values de atención. Debe tener tantas filas como keys. También acepta `image_values`.
    #[serde(alias = "image_values")]
    pub values: Vec<Vec<f64>>,
    /// Distribución previa opcional. Si se omite, se usa uniforme.
    pub prior: Option<Vec<f64>>,
    /// K para métricas top-k. Si se omite, se usa 3 o n, lo que sea menor.
    pub top_k: Option<usize>,
}

impl AttentionConfig {
    pub fn from_yaml_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("No se pudo leer el archivo: {}", path.display()))?;
        let config: AttentionConfig = serde_yaml::from_str(&text)
            .with_context(|| format!("No se pudo parsear YAML de atención: {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        validate_attention_tensors(self.dimension, &self.queries, &self.keys, &self.values)?;
        anyhow::ensure!(self.temperature > 0.0, "temperature debe ser positiva.");
        anyhow::ensure!(
            self.kernel_gamma >= 0.0,
            "kernel_gamma debe ser no negativo."
        );
        self.attention_solver.validate("attention_solver")?;
        validate_prior(&self.prior, self.keys.len())?;
        if let Some(mask) = &self.mask {
            mask.validate(self.queries.len(), self.keys.len())?;
        }
        if let Some(top_k) = self.top_k {
            anyhow::ensure!(top_k > 0, "top_k debe ser mayor que cero.");
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AttentionHeadConfig {
    /// Nombre legible de la cabecera. Si se omite, se usa head_N.
    pub name: Option<String>,
    pub temperature: f64,
    pub kernel_gamma: f64,
    pub prior: Option<Vec<f64>>,
    pub attention_solver: Option<AttentionSolverConfig>,
    pub mask: Option<AttentionMaskConfig>,
}

impl AttentionHeadConfig {
    pub fn validate(&self, index: usize, rows: usize, cols: usize) -> Result<()> {
        anyhow::ensure!(
            self.temperature > 0.0,
            "heads[{}].temperature debe ser positiva.",
            index
        );
        anyhow::ensure!(
            self.kernel_gamma >= 0.0,
            "heads[{}].kernel_gamma debe ser no negativo.",
            index
        );
        validate_prior(&self.prior, cols)?;
        if let Some(solver) = &self.attention_solver {
            solver.validate(&format!("heads[{index}].attention_solver"))?;
        }
        if let Some(mask) = &self.mask {
            mask.validate(rows, cols)?;
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct MultiHeadAttentionConfig {
    pub dimension: usize,
    pub default_attention_solver: AttentionSolverConfig,
    pub default_mask: Option<AttentionMaskConfig>,
    #[serde(alias = "text_queries")]
    pub queries: Vec<Vec<f64>>,
    #[serde(alias = "image_keys")]
    pub keys: Vec<Vec<f64>>,
    #[serde(alias = "image_values")]
    pub values: Vec<Vec<f64>>,
    pub heads: Vec<AttentionHeadConfig>,
    pub top_k: Option<usize>,
}

impl MultiHeadAttentionConfig {
    pub fn from_yaml_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("No se pudo leer el archivo: {}", path.display()))?;
        let config: MultiHeadAttentionConfig = serde_yaml::from_str(&text).with_context(|| {
            format!(
                "No se pudo parsear YAML de multi-head attention: {}",
                path.display()
            )
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        validate_attention_tensors(self.dimension, &self.queries, &self.keys, &self.values)?;
        anyhow::ensure!(
            !self.heads.is_empty(),
            "Debe existir al menos una cabecera."
        );
        self.default_attention_solver
            .validate("default_attention_solver")?;
        if let Some(mask) = &self.default_mask {
            mask.validate(self.queries.len(), self.keys.len())?;
        }
        for (index, head) in self.heads.iter().enumerate() {
            head.validate(index, self.queries.len(), self.keys.len())?;
        }
        if let Some(top_k) = self.top_k {
            anyhow::ensure!(top_k > 0, "top_k debe ser mayor que cero.");
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct AgentSweepConfig {
    pub base_attention: AttentionConfig,
    pub gamma_values: Vec<f64>,
    pub temperature_values: Vec<f64>,
    pub priors: Option<Vec<NamedPriorConfig>>,
    pub objective: AgentObjective,
    pub output_limit: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NamedPriorConfig {
    pub name: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "kebab-case")]
pub enum AgentObjective {
    MaxEntropy,
    MinDistanceToPrior,
    MaxDifferenceFromSoftmax,
    MinOutputShift,
    BalancedTradeoff,
}

impl AgentObjective {
    pub fn as_str(&self) -> &'static str {
        match self {
            AgentObjective::MaxEntropy => "max-entropy",
            AgentObjective::MinDistanceToPrior => "min-distance-to-prior",
            AgentObjective::MaxDifferenceFromSoftmax => "max-difference-from-softmax",
            AgentObjective::MinOutputShift => "min-output-shift",
            AgentObjective::BalancedTradeoff => "balanced-tradeoff",
        }
    }
}

impl AgentSweepConfig {
    pub fn from_yaml_file(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("No se pudo leer el archivo: {}", path.display()))?;
        let config: AgentSweepConfig = serde_yaml::from_str(&text).with_context(|| {
            format!("No se pudo parsear YAML de agent-sweep: {}", path.display())
        })?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        self.base_attention.validate()?;
        anyhow::ensure!(
            !self.gamma_values.is_empty(),
            "gamma_values debe contener al menos un valor."
        );
        anyhow::ensure!(
            !self.temperature_values.is_empty(),
            "temperature_values debe contener al menos un valor."
        );
        anyhow::ensure!(
            self.gamma_values.iter().all(|v| *v >= 0.0),
            "gamma_values debe contener valores no negativos."
        );
        anyhow::ensure!(
            self.temperature_values.iter().all(|v| *v > 0.0),
            "temperature_values debe contener valores positivos."
        );
        if let Some(priors) = &self.priors {
            anyhow::ensure!(
                !priors.is_empty(),
                "priors no puede estar vacío si se define."
            );
            for prior in priors {
                anyhow::ensure!(!prior.name.trim().is_empty(), "Cada prior debe tener name.");
                anyhow::ensure!(
                    prior.values.len() == self.base_attention.keys.len(),
                    "Cada prior debe tener tantas entradas como keys."
                );
                anyhow::ensure!(
                    prior.values.iter().all(|v| *v >= 0.0),
                    "Cada prior debe tener entradas no negativas."
                );
                anyhow::ensure!(
                    prior.values.iter().sum::<f64>() > 0.0,
                    "Cada prior debe tener suma positiva."
                );
            }
        }
        if let Some(limit) = self.output_limit {
            anyhow::ensure!(limit > 0, "output_limit debe ser mayor que cero.");
        }
        Ok(())
    }
}

fn validate_attention_tensors(
    dimension: usize,
    queries: &[Vec<f64>],
    keys: &[Vec<f64>],
    values: &[Vec<f64>],
) -> Result<()> {
    anyhow::ensure!(dimension > 0, "dimension debe ser mayor que cero.");
    anyhow::ensure!(!queries.is_empty(), "Debe existir al menos una query.");
    anyhow::ensure!(!keys.is_empty(), "Debe existir al menos una key.");
    anyhow::ensure!(
        values.len() == keys.len(),
        "values debe tener la misma cantidad de filas que keys."
    );

    for query in queries {
        anyhow::ensure!(
            query.len() == dimension,
            "Cada query debe tener dimensión {}.",
            dimension
        );
    }

    for key in keys {
        anyhow::ensure!(
            key.len() == dimension,
            "Cada key debe tener dimensión {}.",
            dimension
        );
    }

    let value_dimension = values[0].len();
    anyhow::ensure!(value_dimension > 0, "Los values no pueden estar vacíos.");
    for value in values {
        anyhow::ensure!(
            value.len() == value_dimension,
            "Todos los values deben tener la misma dimensión."
        );
    }

    Ok(())
}

fn validate_prior(prior: &Option<Vec<f64>>, expected_len: usize) -> Result<()> {
    if let Some(prior) = prior {
        anyhow::ensure!(
            prior.len() == expected_len,
            "prior debe tener tantas entradas como keys."
        );
        anyhow::ensure!(
            prior.iter().all(|v| *v >= 0.0),
            "prior debe tener entradas no negativas."
        );
        anyhow::ensure!(
            prior.iter().sum::<f64>() > 0.0,
            "prior debe tener suma positiva."
        );
    }
    Ok(())
}
