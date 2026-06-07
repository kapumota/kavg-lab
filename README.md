### KAvgLab

[![CI](https://github.com/kapumota/kavg-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kapumota/kavg-lab/actions/workflows/ci.yml)
![version](https://img.shields.io/badge/version-0.12.0-orange)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Rust](https://img.shields.io/badge/Rust-stable-orange?logo=rust)
![validation](https://img.shields.io/badge/validation-fmt%20%2B%20clippy%20%2B%20tests-brightgreen)
![benchmarks](https://img.shields.io/badge/benchmarks-criterion-blue)
![demo](https://img.shields.io/badge/demo-CLI%20YAML-blue)

KAvgLab es un CLI desarrollado en Rust para experimentar con promedios kernel de funciones convexas, verificación numérica de identidades de Fenchel, comparación de promedios convexos, operadores proximales, solvers de optimización y demostraciones de atención inspiradas en Transformers, LLMs, MLLMs y sistemas de agentes.

El fundamento matemático principal proviene del trabajo **[The kernel average for two convex functions and its application to the extension and representation of monotone operators](https://optimization-online.org/wp-content/uploads/2007/05/1658.pdf)**, de Heinz H. Bauschke y Xianfu Wang. La idea central es combinar funciones convexas mediante una función kernel, lo que permite recuperar y comparar promedios como el promedio aritmético, el promedio epigráfico y el promedio proximal.

El objetivo del proyecto no es implementar un Transformer completo ni entrenar un LLM. Su objetivo es construir un laboratorio CLI, reproducible y auditable para estudiar cómo ideas de análisis convexo, regularización, dualidad de Fenchel, geometría del simplex y atención regularizada pueden conectarse con mecanismos modernos de inteligencia artificial.

#### Estado actual del proyecto

La versión actual del paquete en `Cargo.toml` es:

```text
kavg-lab 0.12.0
```

El proyecto ya incluye:

- CLI en Rust con `clap`.
- Configuración de experimentos mediante YAML.
- Exportación de resultados a CSV.
- Exportación opcional a JSON y manifiesto reproducible para `compute`.
- Funciones convexas, kernels y solvers configurables.
- Demos de atención regularizada, multi-head attention, cross-attention sintética y agent sweep.
- Operadores proximales y verificación Fenchel-Young.
- Suites reproducibles desde CLI.
- Paralelismo determinista opcional con la feature `parallel`.
- Pruebas unitarias, pruebas de integración, property-based testing y benchmarks.
- CI en GitHub Actions para formato, compilación, Clippy, pruebas, build release, feature `parallel` y compilación de benchmarks.
- Validación reproducible centralizada en `scripts/validate.sh` y expuesta mediante `make validate`.

#### Validación del software mediante badges

Los badges superiores no reemplazan a las pruebas. Funcionan como una entrada visual al estado técnico del proyecto:

| Badge | Evidencia asociada | Archivo o comando relacionado |
|---|---|---|
| `CI` | El workflow de GitHub Actions pasa sobre `main` | `.github/workflows/ci.yml` |
| `version` | Versión del paquete Rust | `Cargo.toml` |
| `license` | Licencia del proyecto | `LICENSE` |
| `Rust` | Lenguaje y toolchain esperado | `rust-toolchain.toml` |
| `validation` | Formato, compilación, Clippy y pruebas | `scripts/validate.sh` |
| `benchmarks` | Compilación de benchmarks Criterion | `cargo bench --no-run` |
| `demo` | Ejecución como CLI reproducible con YAML | `examples/` y `experiments/suite.yaml` |

La validación principal debe poder ejecutarse localmente con:

```bash
make validate
```

El badge de CI queda conectado al workflow real. Si `scripts/validate.sh` falla, el workflow falla y el badge deja de indicar un estado correcto.

#### Convenciones del proyecto

- Comentarios, README y mensajes de consola en español.
- Funciones, structs, enums, módulos y métodos en inglés.
- Archivos de configuración en YAML.
- Resultados exportables en CSV para análisis tabular.
- Resultados JSON y manifiestos cuando se requiere auditoría.
- `Cargo.lock` se versiona porque el proyecto es una aplicación CLI y se busca reproducibilidad.
- Los artefactos generados no deben subirse por defecto al repositorio.

#### Qué hace el software

El binario `kavg-lab` expone los siguientes comandos reales:

| Comando | Propósito | Relación conceptual |
|---|---|---|
| `compute` | Calcula el kernel average en puntos definidos por YAML | Análisis convexo y optimización |
| `compare` | Compara promedio aritmético, epigráfico y proximal/kernel average | Comparación de promedios convexos |
| `verify-fenchel` | Verifica numéricamente una identidad de Fenchel para kernel averages | Dualidad convexa |
| `compare-solvers` | Compara varios solvers sobre el mismo experimento | Evaluación algorítmica |
| `prox` | Calcula operadores proximales y, opcionalmente, la envolvente de Moreau | Optimización proximal |
| `fenchel-young` | Verifica la desigualdad de Fenchel-Young | Auditoría matemática local |
| `attention-demo` | Compara atención softmax y atención regularizada por kernel | Transformers y atención regularizada |
| `multihead-attention-demo` | Ejecuta varias cabeceras con priors y parámetros distintos | Multi-head attention |
| `agent-sweep` | Barre hiperparámetros y rankea configuraciones | Agente experimental reproducible |
| `run-suite` | Ejecuta una suite YAML y genera un paquete de evidencia | Reproducibilidad experimental |
| `profile` | Perfila `agent-sweep` y exporta estadísticas de tiempo | Benchmarking CLI |

#### Estructura del repositorio

La organización principal del proyecto es:

```text
kavg-lab/
├── .github/workflows/ci.yml
├── benches/
│   ├── attention_bench.rs
│   ├── kernel_average_bench.rs
│   ├── simplex_projection_bench.rs
│   └── sweep_bench.rs
├── docs/
│   ├── VALIDATION.md
│   ├── fase-1-profesionalizacion-cli.md
│   ├── fase-2-matematica-convexa.md
│   ├── fase-3-solvers-algoritmos.md
│   └── fase-4-experimentacion-cli-reproducible.md
├── examples/
│   ├── functions/
│   ├── attention_demo.yaml
│   ├── attention_causal.yaml
│   ├── attention_custom_mask.yaml
│   ├── attention_sweep.yaml
│   ├── compare_quadratic_l1.yaml
│   ├── cross_attention_multimodal.yaml
│   ├── fase2_*.yaml
│   ├── fase3_*.yaml
│   ├── fase6_*.yaml
│   ├── fenchel_quadratic_l2.yaml
│   ├── multihead_attention.yaml
│   └── quadratic_l1.yaml
├── experiments/
│   ├── README.md
│   └── suite.yaml
├── sample_outputs/
│   └── README.md
├── scripts/
│   └── validate.sh
├── src/
│   ├── attention/
│   ├── fenchel/
│   ├── functions/
│   ├── io/
│   ├── kernels/
│   ├── optimization/
│   ├── prox/
│   ├── cli.rs
│   ├── config.rs
│   ├── lib.rs
│   ├── main.rs
│   ├── math.rs
│   ├── parallel.rs
│   ├── profile.rs
│   └── suite.rs
├── tests/
│   ├── integration_tests.rs
│   ├── property_attention.rs
│   ├── property_kernels.rs
│   ├── property_solvers.rs
│   └── regression_examples.rs
├── CHANGELOG.md
├── CONTRIBUTING.md
├── Cargo.lock
├── Cargo.toml
├── Makefile
├── LICENSE
├── README.md
├── SECURITY.md
└── rust-toolchain.toml
```

#### Instalación de Rust en Linux, macOS o WSL

Instalar dependencias básicas:

```bash
sudo apt update
sudo apt install -y curl build-essential pkg-config
```

Instalar Rust con `rustup`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Activar Cargo en la terminal actual:

```bash
source "$HOME/.cargo/env"
```

Verificar la instalación:

```bash
rustc --version
cargo --version
rustup --version
```

#### Instalación de Rust en Windows

Instalar Rust usando `rustup-init.exe` desde la página oficial de Rust. Durante la instalación, aceptar la configuración por defecto. Si el instalador lo solicita, instalar **Visual Studio C++ Build Tools**.

Abrir una nueva terminal de PowerShell y verificar:

```powershell
rustc --version
cargo --version
rustup --version
```

Si `cargo` no se reconoce, cerrar y volver a abrir PowerShell. Normalmente Rust instala las herramientas en:

```text
%USERPROFILE%\.cargo\bin
```

#### Abrir el proyecto

En Linux, macOS o WSL:

```bash
cd kavg-lab
ls Cargo.toml
```

En Windows PowerShell:

```powershell
cd .\kavg-lab
dir Cargo.toml
```

#### Validación local obligatoria

Antes de presentar, abrir un Pull Request o fusionar a `main`, ejecutar desde la raíz del proyecto:

```bash
make validate
```

Este comando llama a `scripts/validate.sh` y ejecuta la ruta completa de validación:

```text
formato -> compilación -> Clippy -> pruebas -> build release -> feature parallel -> benchmarks
```

También se puede ejecutar el script directamente:

```bash
bash scripts/validate.sh
```

Validación manual equivalente:

```bash
cargo fmt -- --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo build --release
cargo check --all-targets --features parallel
cargo clippy --all-targets --features parallel -- -D warnings
cargo test --all-targets --features parallel
cargo build --release --features parallel
cargo bench --no-run
```

#### CI en GitHub Actions

El workflow `.github/workflows/ci.yml` ejecuta la misma validación centralizada que se usa localmente:

```bash
bash scripts/validate.sh
```

Esto permite demostrar que el proyecto compila, mantiene formato, pasa pruebas, respeta Clippy, soporta la feature opcional `parallel` y compila benchmarks. El badge de CI del README se actualiza a partir del resultado de este workflow.

#### Uso rápido

Ver ayuda general:

```bash
cargo run -- --help
```

Ver ayuda de subcomandos:

```bash
cargo run -- compute --help
cargo run -- compare --help
cargo run -- verify-fenchel --help
cargo run -- compare-solvers --help
cargo run -- prox --help
cargo run -- fenchel-young --help
cargo run -- attention-demo --help
cargo run -- multihead-attention-demo --help
cargo run -- agent-sweep --help
cargo run -- run-suite --help
cargo run -- profile --help
```

#### Análisis convexo básico

Calcular un kernel average:

```bash
cargo run -- compute \
  --config examples/quadratic_l1.yaml \
  --output sample_outputs/results.csv \
  --json sample_outputs/results.json \
  --manifest sample_outputs/manifest.json
```

Comparar promedios convexos:

```bash
cargo run -- compare \
  --config examples/compare_quadratic_l1.yaml \
  --output sample_outputs/comparison.csv
```

Verificar identidad de Fenchel:

```bash
cargo run -- verify-fenchel \
  --config examples/fenchel_quadratic_l2.yaml \
  --output sample_outputs/fenchel.csv
```

#### Uso con OSQP

OSQP se usa como backend alternativo cuando el problema puede expresarse como QP.

```bash
cargo run -- compute \
  --config examples/quadratic_l1_osqp.yaml \
  --output sample_outputs/results_osqp.csv
```

```bash
cargo run -- compare \
  --config examples/compare_quadratic_l1_osqp.yaml \
  --output sample_outputs/comparison_osqp.csv
```

```bash
cargo run -- verify-fenchel \
  --config examples/fenchel_quadratic_l2_osqp.yaml \
  --output sample_outputs/fenchel_osqp.csv
```

#### Operadores proximales y Fenchel-Young

Calcular un operador proximal:

```bash
cargo run -- prox \
  --function examples/functions/l1.yaml \
  --point "[1.0,-2.0,0.5]" \
  --step 0.1 \
  --moreau
```

Verificar Fenchel-Young:

```bash
cargo run -- fenchel-young \
  --function examples/functions/l1.yaml \
  --x "[1.0,-2.0,0.5]" \
  --s "[0.2,-0.1,0.3]"
```

El comando reporta:

```text
f(x)
f*(s)
<x,s>
gap = f(x) + f*(s) - <x,s>
relative_gap
passed
```

#### Comparación de solvers

Comparar varios métodos sobre los mismos puntos:

```bash
cargo run -- compare-solvers \
  --config examples/fase3_compare_solvers.yaml \
  --solvers coordinate-descent,subgradient,osqp,proximal-gradient,fista,admm \
  --output sample_outputs/solver_comparison.csv
```

Solvers convexos soportados:

```text
coordinate-descent
subgradient
osqp
proximal-gradient
fista
admm
```

#### Atención regularizada

Ejecutar demo base de atención:

```bash
cargo run -- attention-demo \
  --config examples/attention_demo.yaml \
  --output sample_outputs/attention_results.csv
```

Ejecutar atención con prior uniforme:

```bash
cargo run -- attention-demo \
  --config examples/attention_demo_uniform.yaml \
  --output sample_outputs/attention_uniform.csv
```

Ejecutar atención causal tipo LLM autoregresivo:

```bash
cargo run -- attention-demo \
  --config examples/attention_causal.yaml \
  --output sample_outputs/attention_causal.csv
```

Ejecutar atención con máscara personalizada:

```bash
cargo run -- attention-demo \
  --config examples/attention_custom_mask.yaml \
  --output sample_outputs/attention_custom_mask.csv
```

Ejecutar atención dispersa con `sparsemax`:

```bash
cargo run -- attention-demo \
  --config examples/fase6_attention_sparsemax.yaml \
  --output sample_outputs/attention_sparsemax.csv
```

Ejecutar atención `top-k`:

```bash
cargo run -- attention-demo \
  --config examples/fase6_attention_topk.yaml \
  --attention-rule top-k \
  --attention-top-k 2 \
  --output sample_outputs/attention_topk.csv
```

Ejecutar atención local tipo sliding window:

```bash
cargo run -- attention-demo \
  --config examples/fase6_attention_local.yaml \
  --output sample_outputs/attention_local.csv
```

Reglas de atención soportadas:

```text
softmax
sparsemax
entmax-1.5
top-k
```

Solvers de atención soportados:

```text
projected-gradient
mirror-descent
frank-wolfe
```

Máscaras soportadas:

```text
none
causal
sliding-window
block-sparse
custom
```

#### Multi-head attention

Ejecutar varias cabeceras de atención con distintos priors, gamma y temperatura:

```bash
cargo run -- multihead-attention-demo \
  --config examples/multihead_attention.yaml \
  --output sample_outputs/multihead_results.csv
```

El CSV resume la salida agregada y métricas de diversidad entre cabeceras:

```text
mean_pairwise_l1
mean_pairwise_l2
mean_pairwise_js
```

#### Cross-attention multimodal sintética

El ejemplo multimodal usa alias semánticos:

```yaml
text_queries: []
image_keys: []
image_values: []
```

Internamente equivalen a:

```yaml
queries: []
keys: []
values: []
```

Ejecutar la demo:

```bash
cargo run -- attention-demo \
  --config examples/cross_attention_multimodal.yaml \
  --output sample_outputs/cross_attention.csv
```

La interpretación es que consultas textuales atienden a regiones visuales sintéticas representadas por embeddings pequeños.

#### Agent sweep

Ejecutar un barrido experimental de hiperparámetros:

```bash
cargo run -- agent-sweep \
  --config examples/attention_sweep.yaml \
  --output sample_outputs/attention_sweep.csv
```

El comando prueba combinaciones de:

```text
gamma
temperature
prior
```

Objetivos disponibles:

```text
max-entropy
min-distance-to-prior
max-difference-from-softmax
min-output-shift
balanced-tradeoff
```

Para `balanced-tradeoff`, el score combina entropía, divergencia frente a softmax, distancia al prior y desplazamiento de salida.

#### Suites reproducibles

Ejecutar una suite completa:

```bash
cargo run -- run-suite \
  --suite experiments/suite.yaml \
  --out evidence/run_001
```

La suite genera una estructura de evidencia como:

```text
evidence/run_001/
├── manifest.json
├── suite.yaml
├── commands.log
├── compute_results.csv
├── fenchel_results.csv
├── attention_results.csv
├── solver_comparison.csv
├── summary.json
└── README.md
```

El directorio `evidence/` está ignorado por Git para evitar subir resultados generados. Si una corrida debe publicarse, conviene empaquetarla y adjuntarla como evidencia externa.

#### Paralelismo determinista opcional

La ejecución secuencial es el comportamiento por defecto. Para usar Rayon, compilar con la feature `parallel` y pasar `--parallel`:

```bash
cargo run --features parallel -- compute \
  --config examples/quadratic_l1.yaml \
  --parallel \
  --jobs auto \
  --output sample_outputs/compute_parallel.csv
```

Comandos con soporte de paralelismo:

```text
compute
verify-fenchel
compare-solvers
attention-demo
multihead-attention-demo
agent-sweep
run-suite
profile
```

Ejemplo de suite reproducible en paralelo:

```bash
cargo run --features parallel -- run-suite \
  --suite experiments/suite.yaml \
  --out evidence/run_parallel_001 \
  --parallel \
  --jobs auto
```

Si el binario se ejecuta con `--parallel` pero fue compilado sin `--features parallel`, el CLI devuelve un error explícito indicando cómo recompilarlo.

#### Profile y benchmarks

Perfilar `agent-sweep` desde el CLI:

```bash
cargo run -- profile \
  --config examples/attention_sweep.yaml \
  --repeat 30 \
  --output sample_outputs/profile.csv
```

Perfilar con paralelismo:

```bash
cargo run --features parallel -- profile \
  --config examples/attention_sweep.yaml \
  --repeat 30 \
  --parallel \
  --jobs auto \
  --output sample_outputs/profile_parallel.csv
```

Ejecutar benchmarks con Criterion:

```bash
cargo bench
```

Columnas generadas por `profile`:

```text
experiment,dimension,n_queries,n_keys,solver,parallel,jobs,repeat,mean_ms,min_ms,max_ms,std_ms
```

#### Funciones convexas soportadas

El proyecto soporta las siguientes funciones convexas en YAML:

```text
quadratic
l1
l2
indicator-box
indicator-simplex
elastic-net
huber
hinge-loss
logistic-loss
max-affine
```

También existen archivos de función independientes en `examples/functions/` para comandos como `prox` y `fenchel-young`.

#### Kernels soportados

Kernels disponibles:

```text
squared-norm
weighted-squared-norm
mahalanobis
huber
entropy-kl
bregman-quadratic
bregman-entropy
```

El kernel de Mahalanobis usa la forma:

```text
g(z) = 1/2 z^T M z
```

La geometría entrópica conecta con KL, mirror descent y atención regularizada sobre el simplex.

#### Fundamento matemático del kernel average

El módulo convexo trabaja con dos funciones convexas `f1` y `f2`, un peso `lambda1` y un kernel `g`.

Pesos convexos:

```text
lambda1 in (0, 1)
lambda2 = 1 - lambda1
```

Restricción de mezcla:

```text
lambda1 * y1 + lambda2 * y2 = x
```

El código elimina `y2` usando:

```text
y2 = (x - lambda1 * y1) / lambda2
```

Para el kernel cuadrático:

```text
g(z) = 1/2 ||z||²
z = y1 - y2
```

El promedio kernel se calcula como:

```text
P(x) = min_{y1,y2} lambda1*f1(y1)
                    + lambda2*f2(y2)
                    + c*lambda1*lambda2*g(y1-y2)

sujeto a: lambda1*y1 + lambda2*y2 = x
```

El factor `c` permite comparar:

```text
c = 0 -> promedio epigráfico
c = 1 -> proximal/kernel average con squared-norm
```

El promedio aritmético usado como referencia es:

```text
A(x) = lambda1*f1(x) + lambda2*f2(x)
```

#### Identidad de Fenchel

El comando `verify-fenchel` verifica numéricamente una identidad de dualidad:

```text
(P(f1, f2, g))* (s) ≈ P(f1*, f2*, g*) (s)
```

El lado izquierdo se aproxima por maximización:

```text
sup_x <s, x> - P(f1, f2, g)(x)
```

El lado derecho se calcula usando conjugados analíticos disponibles:

```text
f1*, f2*, g*
```

Para el kernel cuadrático:

```text
g(z) = 1/2 ||z||²
g*(s) = 1/2 ||s||²
```

El CSV exporta valor izquierdo aproximado, valor derecho, error absoluto, error relativo y estado `passed`.

#### Atención regularizada como optimización

Para una query `q` y keys `k_i`, el score se calcula como:

```text
score_i = <q, k_i> / sqrt(d)
```

La atención regularizada resuelve un problema sobre el simplex:

```text
min_p  - <scores, p>
       + temperature * sum_i p_i log(p_i)
       + gamma/2 * ||p - p0||²

sujeto a:
       p_i >= 0
       sum_i p_i = 1
       p_i = 0 en posiciones bloqueadas
```

Interpretación:

- `-<scores,p>` favorece tokens con score alto.
- `temperature * sum p_i log(p_i)` introduce suavidad entrópica.
- `gamma/2 * ||p-p0||²` acerca la atención al prior estructural.
- La restricción `sum_i p_i = 1` mantiene una distribución de probabilidad.
- Las máscaras causales o estructuradas imponen restricciones similares a las usadas en modelos autoregresivos y atención local.

La salida se calcula como:

```text
output = sum_i p_i * value_i
```

#### Salidas generadas y limpieza

El repositorio ignora por defecto:

```text
/target
*.csv
sample_outputs/*.csv
sample_outputs/*.json
/evidence/
```

Esto evita subir binarios, resultados temporales y paquetes de evidencia generados localmente.

Para limpiar resultados temporales comunes:

```bash
make clean
```

Comandos equivalentes:

```bash
rm -rf target
rm -rf evidence
find . -type f -name "*.csv" -delete
find sample_outputs -type f -name "*.json" -delete
```

Revisar antes de confirmar cambios:

```bash
git status
git diff -- README.md Cargo.toml .github/workflows/ci.yml
```

#### Fases implementadas o reflejadas en el proyecto

| Fase | Estado reflejado | Evidencia principal |
|---|---|---|
| Fase 1 | Profesionalización del CLI | CI, `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`, JSON y manifiesto en `compute` |
| Fase 2 | Matemática convexa ampliada | Funciones convexas y kernels adicionales en `src/functions/`, `src/kernels/` y ejemplos `fase2_*.yaml` |
| Fase 3 | Solvers y comparación algorítmica | `compare-solvers`, solvers convexos y solvers de atención |
| Fase 4 | Experimentación reproducible | `run-suite`, `experiments/suite.yaml`, paquetes en `evidence/` |
| Fase 5 | Paralelismo determinista | `src/parallel.rs`, feature `parallel`, flags `--parallel` y `--jobs` |
| Fase 6 | Atención dispersa | `sparsemax`, `entmax-1.5`, `top-k`, `sliding-window`, `block-sparse` |
| Fase 7 | Matemática convexa auditable | `prox`, `fenchel-young`, `bregman-entropy` |
| Fase 8 | Pruebas de propiedades y benchmarking | `proptest`, `benches/`, `profile` |

Observación importante: los documentos individuales en `docs/` cubren formalmente las fases 1 a 4. Las fases posteriores están reflejadas en código, ejemplos, tests, benchmarks y el registro de cambios, aunque no todas tienen todavía un documento independiente en `docs/`.

#### Flujo recomendado con ramas y Pull Request

Crear una rama para ordenar documentación:

```bash
git switch -c fase-documentacion-readme-ordenado
```

Reemplazar `README.md` y validar:

```bash
make validate
```

Confirmar cambios:

```bash
git add README.md
git commit -m "Ordena README segun estado actual del proyecto"
git push -u origin fase-documentacion-readme-ordenado
```

Abrir un Pull Request hacia `main` y esperar que el CI pase antes de fusionar.

#### Alcance y limitaciones

KAvgLab es un laboratorio CLI de optimización convexa y atención regularizada, no como un framework de deep learning completo. Su valor está en que cada resultado se obtiene desde configuraciones YAML pequeñas, comandos reproducibles y salidas auditables.

El proyecto es adecuado para:

- experimentos reproducibles sobre promedios kernel,
- comparación de solvers,
- demostraciones interpretables de atención regularizada,
- conexión conceptual entre optimización, Transformers, LLMs, MLLMs y agentes.

No pretende reemplazar a PyTorch, JAX, Hugging Face Transformers ni frameworks industriales de entrenamiento de modelos.
