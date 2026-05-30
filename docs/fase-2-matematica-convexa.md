### Fase 2: matemática convexa ampliada

Esta fase mantiene a KAvgLab como una herramienta de línea de comandos. No agrega dashboard, interfaz gráfica ni servidor web. El objetivo es ampliar el núcleo matemático para que el CLI pueda modelar más funciones convexas, más kernels y más geometrías de optimización.

#### Objetivo técnico

La Fase 1 profesionalizó el repositorio y la trazabilidad de ejecución. La Fase 2 fortalece el contenido matemático del software:

- Más funciones convexas configurables desde YAML.
- Más kernels convexos.
- Kernel de Mahalanobis para geometrías no euclidianas.
- Soporte OSQP ampliado para casos cuadráticos con L1, cajas y simplex.
- Conjugado analítico de L1 como indicador de la bola infinito.

#### Funciones convexas agregadas

Además de `quadratic`, `l1` y `l2`, ahora el archivo YAML puede usar:

```text
indicator-box
indicator-simplex
elastic-net
huber
hinge-loss
logistic-loss
max-affine
```

Estas funciones permiten pasar de ejemplos mínimos a experimentos de optimización más cercanos a aprendizaje automático, regularización, restricciones convexas y modelos por piezas.

#### Kernels agregados

Además de `squared-norm`, se agregan:

```text
weighted-squared-norm
mahalanobis
huber
entropy-kl
bregman-quadratic
```

El kernel más importante de esta fase es `mahalanobis`, definido como:

```text
g(z) = 1/2 z^T M z
```

Este kernel permite estudiar geometrías distintas a la euclidiana. En lugar de medir todos los ejes con el mismo peso, la matriz `M` controla qué direcciones son más costosas o más relevantes. Esto conecta el proyecto con métricas aprendidas, embeddings, atención regularizada y optimización convexa.

#### Ejemplo principal

```yaml
dimension: 3
lambda1: 0.5

f1:
  type: elastic-net
  l1_alpha: 0.1
  l2_alpha: 1.0

f2:
  type: indicator-box
  lower: [-1.0, -1.0, -1.0]
  upper: [1.0, 1.0, 1.0]

kernel:
  type: mahalanobis
  matrix:
    - [2.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.5]

solver:
  method: osqp
  initial_step: 0.1
  tolerance: 1.0e-8
  min_step: 1.0e-10
  max_iterations: 10000

points:
  - [0.5, 0.2, -0.1]
  - [1.0, -0.5, 0.3]
```

Ejecución recomendada:

```bash
cargo run -- compute \
  --config examples/fase2_elastic_box_mahalanobis.yaml \
  --output sample_outputs/fase2_results.csv \
  --json sample_outputs/fase2_results.json \
  --manifest sample_outputs/fase2_manifest.json
```

Los archivos generados dentro de `sample_outputs/` siguen ignorados por Git, salvo `sample_outputs/README.md`.

#### Alcance de OSQP en esta fase

OSQP se usa para casos que pueden expresarse como QP:

```text
quadratic
l2
l1
elastic-net
indicator-box
indicator-simplex
squared-norm
weighted-squared-norm
mahalanobis
bregman-quadratic
```

Las funciones `huber`, `hinge-loss`, `logistic-loss`, `max-affine` y el kernel `entropy-kl` se pueden usar con métodos de primer orden como `subgradient` o `coordinate-descent`, pero no se envían a OSQP en esta fase.

#### Validación local

Antes de mezclar esta fase a `main`, ejecutar:

```bash
cargo fmt -- --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo build --release
```

También conviene probar:

```bash
cargo run -- compute --config examples/fase2_elastic_box_mahalanobis.yaml
cargo run -- compute --config examples/fase2_simplex_weighted.yaml
cargo run -- compute --config examples/fase2_losses_huber.yaml
```
