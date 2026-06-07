### Validación de KAvgLab

#### Objetivo

Este documento explica cómo se valida que KAvgLab funciona como software reproducible y no solo como una colección de ejemplos.

#### Fuente principal de validación

La validación está centralizada en:

```bash
scripts/validate.sh
```

El comando recomendado para ejecutarla es:

```bash
make validate
```

#### Qué se valida

El script ejecuta las siguientes comprobaciones:

| Etapa | Comando | Propósito |
|---|---|---|
| Toolchain | `rustc --version` y `cargo --version` | Confirmar entorno Rust disponible |
| Formato | `cargo fmt -- --check` | Verificar estilo consistente |
| Compilación | `cargo check --all-targets` | Validar todos los targets sin build completo |
| Lint | `cargo clippy --all-targets -- -D warnings` | Rechazar advertencias importantes |
| Pruebas | `cargo test --all-targets` | Ejecutar pruebas unitarias, integración y propiedades |
| Release | `cargo build --release` | Verificar binario optimizado |
| Paralelismo | `cargo check`, `clippy`, `test` y `build` con `--features parallel` | Verificar la feature opcional |
| Benchmarks | `cargo bench --no-run` | Confirmar que los benchmarks compilan |

#### Relación con los badges

El README muestra badges de estado, versión, licencia, lenguaje, validación, benchmarks y demo. El badge más importante es `CI`, porque está conectado al workflow real:

```text
.github/workflows/ci.yml
```

Ese workflow ejecuta:

```bash
bash scripts/validate.sh
```

Si la validación falla, el workflow falla y el badge de CI deja de mostrar un estado correcto.

#### Validación local antes de un Pull Request

Antes de abrir un Pull Request hacia `main`, ejecutar:

```bash
make clean
make validate
```

Después revisar el estado del repositorio:

```bash
git status --short
```

El repositorio no debe incluir binarios, salidas CSV, salidas JSON temporales, paquetes de evidencia ni directorios `target/`.
