use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobCount {
    Auto,
    Fixed(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Sequential,
    Parallel { jobs: JobCount },
}

impl ExecutionMode {
    pub fn is_parallel(self) -> bool {
        matches!(self, ExecutionMode::Parallel { .. })
    }

    pub fn label(self) -> &'static str {
        match self {
            ExecutionMode::Sequential => "sequential",
            ExecutionMode::Parallel { .. } => "parallel",
        }
    }

    pub fn cli_suffix(self) -> String {
        match self {
            ExecutionMode::Sequential => String::new(),
            ExecutionMode::Parallel { jobs } => match jobs {
                JobCount::Auto => " --parallel --jobs auto".to_string(),
                JobCount::Fixed(value) => format!(" --parallel --jobs {value}"),
            },
        }
    }
}

pub fn parse_execution_mode(parallel: bool, jobs: &str) -> Result<ExecutionMode> {
    if !parallel {
        return Ok(ExecutionMode::Sequential);
    }

    let jobs = parse_jobs(jobs)?;

    #[cfg(not(feature = "parallel"))]
    {
        let _ = jobs;
        bail!(
            "El binario fue compilado sin la feature `parallel`. Recompile con: cargo run --features parallel -- ..."
        );
    }

    #[cfg(feature = "parallel")]
    {
        configure_rayon(jobs)?;
        Ok(ExecutionMode::Parallel { jobs })
    }
}

fn parse_jobs(value: &str) -> Result<JobCount> {
    let normalized = value.trim().to_ascii_lowercase();
    if normalized == "auto" {
        return Ok(JobCount::Auto);
    }

    let parsed = normalized
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("--jobs debe ser `auto` o un entero positivo."))?;
    if parsed == 0 {
        bail!("--jobs debe ser mayor que cero.");
    }
    Ok(JobCount::Fixed(parsed))
}

#[cfg(feature = "parallel")]
fn configure_rayon(jobs: JobCount) -> Result<()> {
    if let JobCount::Fixed(value) = jobs {
        match rayon::ThreadPoolBuilder::new()
            .num_threads(value)
            .build_global()
        {
            Ok(()) => {}
            Err(_) => {
                // El pool global de Rayon solo puede inicializarse una vez por proceso.
                // Si ya existe, se conserva para mantener compatibilidad con suites largas.
            }
        }
    }
    Ok(())
}

pub fn map_indexed<T, R, F>(items: &[T], mode: ExecutionMode, f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> Result<R> + Sync + Send,
{
    match mode {
        ExecutionMode::Sequential => items
            .iter()
            .enumerate()
            .map(|(index, item)| f(index, item))
            .collect(),
        ExecutionMode::Parallel { .. } => map_indexed_parallel(items, f),
    }
}

#[cfg(feature = "parallel")]
fn map_indexed_parallel<T, R, F>(items: &[T], f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> Result<R> + Sync + Send,
{
    use rayon::prelude::*;

    items
        .par_iter()
        .enumerate()
        .map(|(index, item)| f(index, item))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn map_indexed_parallel<T, R, F>(_items: &[T], _f: F) -> Result<Vec<R>>
where
    T: Sync,
    R: Send,
    F: Fn(usize, &T) -> Result<R> + Sync + Send,
{
    bail!(
        "El binario fue compilado sin la feature `parallel`. Recompile con: cargo run --features parallel -- ..."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_mode_ignores_jobs() {
        assert_eq!(
            parse_execution_mode(false, "auto").unwrap(),
            ExecutionMode::Sequential
        );
        assert_eq!(
            parse_execution_mode(false, "8").unwrap(),
            ExecutionMode::Sequential
        );
    }
}
