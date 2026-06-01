### Fase 1: profesionalización del CLI sin cambiar su esencia

Esta fase conserva a KAvgLab como una herramienta de línea de comandos en Rust. No introduce dashboard, interfaz web ni visualizaciones obligatorias. La mejora se concentra en estructura de repositorio, validación automática, trazabilidad y salidas reproducibles.

#### Objetivos

- Mantener el uso principal desde terminal.
- Conservar la salida CSV existente.
- Agregar salida JSON opcional para auditoría automática.
- Agregar manifiesto de ejecución opcional para trazabilidad.
- Agregar CI básico para validar formato, compilación, linting, pruebas y build release.
- Agregar documentación mínima de mantenimiento del proyecto.

#### Estructura agregada

```text
.github/workflows/ci.yml
CHANGELOG.md
CONTRIBUTING.md
SECURITY.md
docs/
experiments/
sample_outputs/
LICENSE
```

#### Flujo local recomendado

Como el proyecto ya existe en GitHub y se trabaja localmente, la fase se debe aplicar en una rama separada:

```bash
git switch -c fase-1-profesionalizacion-cli
```

Luego se validan los cambios:

```bash
cargo fmt -- --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo build --release
```

Si todo pasa correctamente:

```bash
git status
git add .
git commit -m "Agrega fase 1 de profesionalizacion del CLI"
git push -u origin fase-1-profesionalizacion-cli
```

Después se abre un Pull Request hacia `main`.

## Uso esperado de `compute`

La salida CSV anterior se mantiene:

```bash
kavg-lab compute \
  --config examples/quadratic_l1.yaml \
  --output sample_outputs/results.csv
```

Además, ahora puede generarse JSON estructurado:

```bash
kavg-lab compute \
  --config examples/quadratic_l1.yaml \
  --output sample_outputs/results.csv \
  --json sample_outputs/results.json
```

Y también un manifiesto reproducible:

```bash
kavg-lab compute \
  --config examples/quadratic_l1.yaml \
  --output sample_outputs/results.csv \
  --json sample_outputs/results.json \
  --manifest sample_outputs/manifest.json
```

#### Archivos de salida

- `results.csv`: salida tabular compatible con el flujo anterior.
- `results.json`: salida estructurada para auditoría y comparación automática.
- `manifest.json`: metadatos de ejecución, versión del binario, ruta de configuración, hash del archivo YAML, cantidad de resultados y duración.

#### Artefactos que no se deben subir

Los archivos generados en `sample_outputs/` no deben subirse salvo que sean ejemplos pequeños y deliberados. Por defecto, `.gitignore` excluye:

```text
sample_outputs/*.csv
sample_outputs/*.json
```

El archivo que sí se mantiene versionado es:

```text
sample_outputs/README.md
```
