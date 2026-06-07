#!/usr/bin/env bash
set -euo pipefail

run_step() {
  local title="$1"
  shift
  echo
  echo "==> ${title}"
  "$@"
}

require_file() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "Falta el archivo requerido: ${file}"
    exit 1
  fi
}

echo "Validando KAvgLab"

require_file "Cargo.toml"
require_file "Cargo.lock"
require_file "README.md"
require_file "LICENSE"
require_file "rust-toolchain.toml"

if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo
  echo "==> Verificando que no existan artefactos generados versionados"
  if git ls-files | grep -E '(^target/|^evidence/|__pycache__|\.pyc$|\.pyo$|\.csv$|sample_outputs/.*\.json$)' >/tmp/kavg-lab-generated-files.txt; then
    echo "Se encontraron artefactos generados versionados:"
    cat /tmp/kavg-lab-generated-files.txt
    exit 1
  fi
fi

run_step "Mostrando toolchain" rustc --version
run_step "Mostrando Cargo" cargo --version
run_step "Verificando formato" cargo fmt -- --check
run_step "Compilando todos los targets" cargo check --all-targets
run_step "Ejecutando Clippy" cargo clippy --all-targets -- -D warnings
run_step "Ejecutando pruebas" cargo test --all-targets
run_step "Construyendo release" cargo build --release
run_step "Compilando con feature parallel" cargo check --all-targets --features parallel
run_step "Ejecutando Clippy con feature parallel" cargo clippy --all-targets --features parallel -- -D warnings
run_step "Ejecutando pruebas con feature parallel" cargo test --all-targets --features parallel
run_step "Construyendo release con feature parallel" cargo build --release --features parallel
run_step "Compilando benchmarks" cargo bench --no-run

echo
echo "Validacion completada correctamente"
