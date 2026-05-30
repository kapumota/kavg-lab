use crate::optimization::kernel_average::KernelAverageResult;
use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Entrada mínima para escribir un manifiesto de ejecución sin agregar dependencias pesadas.
pub struct ExecutionManifest<'a> {
    pub command: &'a str,
    pub config_path: &'a Path,
    pub csv_output: Option<&'a Path>,
    pub json_output: Option<&'a Path>,
    pub started_at: SystemTime,
    pub finished_at: SystemTime,
    pub result_count: usize,
    pub status: &'a str,
}

/// Exporta los resultados de `compute` a JSON estructurado.
pub fn export_compute_results_json(path: &Path, results: &[KernelAverageResult]) -> Result<()> {
    ensure_parent_dir(path)?;

    let mut text = String::new();
    text.push_str("{\n");
    text.push_str(&format!(
        "  \"schema_version\": \"1.0\",\n  \"tool\": {{\n    \"name\": \"kavg-lab\",\n    \"version\": {}\n  }},\n  \"command\": \"compute\",\n  \"result_count\": {},\n  \"results\": [\n",
        json_string(env!("CARGO_PKG_VERSION")),
        results.len()
    ));

    for (i, result) in results.iter().enumerate() {
        text.push_str("    {\n");
        text.push_str(&format!(
            "      \"index\": {},\n      \"point\": {},\n      \"average_kind\": {},\n      \"value\": {},\n      \"raw_penalty\": {},\n      \"weighted_penalty\": {},\n      \"iterations\": {},\n      \"solver_method\": {},\n      \"solver_metric\": {},\n      \"y1\": {},\n      \"y2\": {}\n",
            result.index.unwrap_or_default(),
            option_vec_to_json(result.point.as_ref()),
            json_string(&result.average_kind),
            json_number(result.value),
            json_number(result.raw_penalty),
            json_number(result.weighted_penalty),
            result.iterations,
            json_string(&result.solver_method),
            json_number(result.solver_metric),
            vec_to_json(&result.y1),
            vec_to_json(&result.y2)
        ));
        if i + 1 == results.len() {
            text.push_str("    }\n");
        } else {
            text.push_str("    },\n");
        }
    }

    text.push_str("  ]\n}\n");
    fs::write(path, text).with_context(|| format!("No se pudo escribir JSON: {}", path.display()))
}

/// Exporta un manifiesto de ejecución en JSON.
pub fn export_execution_manifest(path: &Path, manifest: &ExecutionManifest<'_>) -> Result<()> {
    ensure_parent_dir(path)?;

    let config_hash = fnv1a64_file(manifest.config_path)?;
    let elapsed_ms = manifest
        .finished_at
        .duration_since(manifest.started_at)
        .unwrap_or(Duration::from_millis(0))
        .as_millis();

    let text = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"1.0\",\n",
            "  \"tool\": {{\n",
            "    \"name\": \"kavg-lab\",\n",
            "    \"version\": {}\n",
            "  }},\n",
            "  \"command\": {},\n",
            "  \"config_path\": {},\n",
            "  \"config_hash_fnv1a64\": {},\n",
            "  \"csv_output\": {},\n",
            "  \"json_output\": {},\n",
            "  \"started_at_unix_ms\": {},\n",
            "  \"finished_at_unix_ms\": {},\n",
            "  \"elapsed_ms\": {},\n",
            "  \"result_count\": {},\n",
            "  \"git_commit\": {},\n",
            "  \"status\": {}\n",
            "}}\n"
        ),
        json_string(env!("CARGO_PKG_VERSION")),
        json_string(manifest.command),
        json_string(&manifest.config_path.display().to_string()),
        json_string(&format!("{config_hash:016x}")),
        optional_path_to_json(manifest.csv_output),
        optional_path_to_json(manifest.json_output),
        unix_millis(manifest.started_at),
        unix_millis(manifest.finished_at),
        elapsed_ms,
        manifest.result_count,
        json_string(option_env!("GIT_COMMIT").unwrap_or("unknown")),
        json_string(manifest.status)
    );

    fs::write(path, text)
        .with_context(|| format!("No se pudo escribir manifiesto: {}", path.display()))
}

fn ensure_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!("No se pudo crear el directorio padre: {}", parent.display())
            })?;
        }
    }
    Ok(())
}

fn fnv1a64_file(path: &Path) -> Result<u64> {
    let bytes = fs::read(path)
        .with_context(|| format!("No se pudo leer el archivo para hash: {}", path.display()))?;
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(hash)
}

fn optional_path_to_json(path: Option<&Path>) -> String {
    match path {
        Some(path) => json_string(&path.display().to_string()),
        None => "null".to_string(),
    }
}

fn option_vec_to_json(values: Option<&Vec<f64>>) -> String {
    match values {
        Some(values) => vec_to_json(values),
        None => "null".to_string(),
    }
}

fn vec_to_json(values: &[f64]) -> String {
    let items = values
        .iter()
        .map(|value| json_number(*value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{items}]")
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.12}")
    } else if value.is_nan() {
        json_string("NaN")
    } else if value.is_sign_positive() {
        json_string("Infinity")
    } else {
        json_string("-Infinity")
    }
}

fn json_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            c if c.is_control() => escaped.push_str(&format!("\\u{:04x}", c as u32)),
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}

fn unix_millis(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_millis(0))
        .as_millis()
}
