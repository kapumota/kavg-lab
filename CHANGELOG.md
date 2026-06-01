# Registro de cambios

## [0.8.0] - Fase 4: experimentación CLI reproducible

### Agregado

- Comando `run-suite` para ejecutar suites reproducibles desde YAML.
- Generación de paquetes de evidencia con `manifest.json`, `summary.json`, `commands.log`, copia de `suite.yaml` y resultados CSV.
- Suite de referencia en `experiments/suite.yaml` para `compute`, `verify-fenchel`, `attention-demo` y `compare-solvers`.
- Documento `docs/fase-4-experimentacion-cli-reproducible.md`.

### Mantenido

- El proyecto sigue siendo un CLI puro.
- No se agrega dashboard, servidor web ni interfaz gráfica.


## [0.7.0] - Fase 3: solvers y comparación algorítmica

### Agregado

- Nuevos métodos para promedios kernel: `proximal-gradient`, `fista` y `admm`.
- Nuevos métodos para atención sobre simplex: `mirror-descent` y `frank-wolfe`, manteniendo `projected-gradient`.
- Nuevo comando `compare-solvers` para comparar varios métodos sobre el mismo YAML.
- Opción `--solver` en `attention-demo` para sobrescribir el solver del YAML desde CLI.
- Documentación de Fase 3 en `docs/fase-3-solvers-algoritmos.md`.
- Ejemplos YAML para comparación de solvers y atención con mirror descent.

### Mantenido

- El proyecto sigue siendo un CLI puro.
- No se agrega dashboard, servidor ni interfaz gráfica.

## [0.6.0] - Fase 2: matematica convexa ampliada

### Agregado

- Nuevas funciones convexas: `indicator-box`, `indicator-simplex`, `elastic-net`, `huber`, `hinge-loss`, `logistic-loss` y `max-affine`.
- Nuevos kernels: `weighted-squared-norm`, `mahalanobis`, `huber`, `entropy-kl` y `bregman-quadratic`.
- Soporte OSQP ampliado para elastic-net, restricciones de caja, restricciones de simplex y kernels cuadráticos generales.
- Conjugado de L1 como indicador de la bola infinito para verificaciones de Fenchel compatibles.
- Ejemplos YAML de Fase 2 en `examples/`.
- Documento `docs/fase-2-matematica-convexa.md`.

### Mantenido

- El proyecto sigue siendo un CLI puro en Rust.
- No se agregan dashboard, interfaz gráfica ni servidor web.


Todos los cambios notables de este proyecto se documentarán en este archivo.

El formato sigue una convención simple inspirada en Keep a Changelog y versionado semántico.

## [0.5.0] - Sin publicar

### Agregado

- Flujo de CI para Rust en GitHub Actions.
- Documentos base de mantenimiento: `CONTRIBUTING.md`, `SECURITY.md` y `CHANGELOG.md`.
- Directorios versionables para documentación, experimentos y salidas de ejemplo.
- Soporte inicial para exportar resultados de `compute` en JSON sin eliminar la salida CSV.
- Soporte inicial para generar un manifiesto de ejecución de `compute` con metadatos reproducibles.

### Cambiado

- Renombrado `LICENCE` a `LICENSE` para seguir la convención más reconocida por GitHub y herramientas externas.

## [0.4.2] - Línea base

### Agregado

- CLI para promedios kernel, comparación de promedios convexos, verificación de Fenchel y demostraciones de atención regularizada.
- Ejemplos YAML para experimentos convexos, atención, multi-head attention, cross-attention sintético y agent sweep.
- Exportación CSV para los principales comandos.
