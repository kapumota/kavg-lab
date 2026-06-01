use clap::{Parser, Subcommand};
use kavg_lab::config::AttentionSolverMethod;
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
    },

    /// Verifica numéricamente la identidad de Fenchel del kernel average.
    VerifyFenchel {
        /// Ruta del archivo de configuración YAML.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar la verificación en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Ejecuta una demostracion de atención softmax y atención regularizada por kernel.
    AttentionDemo {
        /// Ruta del archivo de configuración YAML de atención.
        #[arg(short, long)]
        config: PathBuf,

        /// Sobrescribe el solver del YAML: projected-gradient, mirror-descent o frank-wolfe.
        #[arg(long)]
        solver: Option<AttentionSolverMethod>,

        /// Ruta opcional para exportar los resultados en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Ejecuta varias cabeceras de atención con distintos priors, gamma o temperatura.
    MultiheadAttentionDemo {
        /// Ruta del archivo de configuración YAML multi-head.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar los resultados en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Ejecuta una suite reproducible y genera un paquete de evidencia CLI.
    RunSuite {
        /// Ruta del archivo YAML de suite reproducible.
        #[arg(long)]
        suite: PathBuf,

        /// Directorio de salida para el paquete de evidencia.
        #[arg(long)]
        out: PathBuf,
    },

    /// Ejecuta un barrido experimental sobre gamma, temperature y priors.
    AgentSweep {
        /// Ruta del archivo de configuración YAML del barrido.
        #[arg(short, long)]
        config: PathBuf,

        /// Ruta opcional para exportar los resultados en CSV.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}
