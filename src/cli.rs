use clap::{Parser, Subcommand};
use kavg_lab::config::{AttentionRule, AttentionSolverMethod};
use std::path::PathBuf;

/// CLI  para experimentar con promedios kernel, OSQP y atención regularizada.
#[derive(Parser, Debug)]
#[command(name = "kavg-lab")]
#[command(version)]
#[command(
    about = "Calcula, compara y verifica promedios kernel; incluye demos tipo Transformer/IA"
)]
#[command(disable_help_subcommand = true)]
#[command(help_template = "\
{about}

Uso:
  {usage}

Comandos:
{subcommands}

Opciones:
{options}
")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Calcula el kernel average en los puntos definidos por un archivo YAML.
    Compute {
        /// Ruta del archivo de configuración YAML.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar resultados en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Ruta opcional para exportar resultados estructurados en JSON.
        #[arg(long = "json")]
        json_output: Option<PathBuf>,

        /// Ruta opcional para exportar un manifiesto reproducible de la ejecución.
        #[arg(long)]
        manifest: Option<PathBuf>,

        /// Ejecuta los puntos de forma paralela si el binario fue compilado con --features parallel.
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Número de workers para Rayon: auto o entero positivo.
        #[arg(long, default_value = "auto")]
        jobs: String,
    },

    /// Calcula operadores proximales y, opcionalmente, el valor de Moreau.
    Prox {
        /// YAML de función convexa independiente.
        #[arg(long)]
        function: PathBuf,

        /// Punto en formato YAML/JSON, por ejemplo: "[1.0,-2.0,0.5]".
        #[arg(long)]
        point: String,

        /// Paso proximal t > 0.
        #[arg(long)]
        step: f64,

        /// Muestra explícitamente el valor y gradiente de la envolvente de Moreau.
        #[arg(long, default_value_t = false)]
        moreau: bool,
    },

    /// Verifica la desigualdad de Fenchel-Young f(x)+f*(s) >= <x,s>.
    FenchelYoung {
        /// YAML de función convexa independiente.
        #[arg(long)]
        function: PathBuf,

        /// Punto primal x en formato YAML/JSON.
        #[arg(long)]
        x: String,

        /// Punto dual s en formato YAML/JSON.
        #[arg(long)]
        s: String,

        /// Tolerancia para decidir passed.
        #[arg(long, default_value_t = 1.0e-8)]
        tolerance: f64,
    },

    /// Compara promedio aritmético, epigráfico y proximal average.
    Compare {
        /// Ruta del archivo de configuración YAML.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar la comparación en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Compara varios solvers sobre el mismo experimento YAML.
    CompareSolvers {
        /// Ruta del archivo de configuración YAML.
        #[arg(short, long)]
        config: PathBuf,

        /// Lista separada por comas: coordinate-descent,subgradient,osqp,proximal-gradient,fista,admm.
        #[arg(long, value_delimiter = ',')]
        solvers: Vec<String>,

        /// Ruta opcional para exportar la comparación en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Ejecuta combinaciones solver × punto en paralelo si está disponible la feature parallel.
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Número de workers para Rayon: auto o entero positivo.
        #[arg(long, default_value = "auto")]
        jobs: String,
    },

    /// Verifica numéricamente la identidad de Fenchel del kernel average.
    VerifyFenchel {
        /// Ruta del archivo de configuración YAML.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar la verificación en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Ejecuta puntos duales en paralelo si está disponible la feature parallel.
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Número de workers para Rayon: auto o entero positivo.
        #[arg(long, default_value = "auto")]
        jobs: String,
    },

    /// Ejecuta una demostracion de atención softmax y atención regularizada por kernel.
    AttentionDemo {
        /// Ruta del archivo de configuración YAML de atención.
        #[arg(short, long)]
        config: PathBuf,

        /// Sobrescribe el solver del YAML: projected-gradient, mirror-descent o frank-wolfe.
        #[arg(long)]
        solver: Option<AttentionSolverMethod>,

        /// Sobrescribe la regla base de atención: softmax, sparsemax, entmax-1.5 o top-k.
        #[arg(long = "attention-rule")]
        attention_rule: Option<AttentionRule>,

        /// K usado cuando --attention-rule top-k.
        #[arg(long = "attention-top-k")]
        attention_top_k: Option<usize>,

        /// Ruta opcional para exportar los resultados en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Ejecuta queries independientes en paralelo si está disponible la feature parallel.
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Número de workers para Rayon: auto o entero positivo.
        #[arg(long, default_value = "auto")]
        jobs: String,
    },

    /// Ejecuta varias cabeceras de atención con distintos priors, gamma o temperatura.
    MultiheadAttentionDemo {
        /// Ruta del archivo de configuración YAML multi-head.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar los resultados en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Ejecuta queries/cabeceras independientes en paralelo si está disponible la feature parallel.
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Número de workers para Rayon: auto o entero positivo.
        #[arg(long, default_value = "auto")]
        jobs: String,
    },

    /// Ejecuta una suite reproducible y genera un paquete de evidencia CLI.
    RunSuite {
        /// Ruta del archivo YAML de suite reproducible.
        #[arg(long)]
        suite: PathBuf,

        /// Directorio de salida para el paquete de evidencia.
        #[arg(long)]
        out: PathBuf,

        /// Ejecuta pasos internos paralelizables si está disponible la feature parallel.
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Número de workers para Rayon: auto o entero positivo.
        #[arg(long, default_value = "auto")]
        jobs: String,
    },

    /// Ejecuta un barrido experimental sobre gamma, temperature y priors.
    AgentSweep {
        /// Ruta del archivo de configuración YAML del barrido.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar los resultados en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Ejecuta configuraciones independientes del sweep en paralelo si está disponible la feature parallel.
        #[arg(long, default_value_t = false)]
        parallel: bool,

        /// Número de workers para Rayon: auto o entero positivo.
        #[arg(long, default_value = "auto")]
        jobs: String,
    },
}
