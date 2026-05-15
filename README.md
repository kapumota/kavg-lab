### KAvgLab 

KAvgLab es un CLI  desarrollado en Rust para experimentar con promedios kernel de funciones convexas, verificación numérica de identidades de Fenchel, comparación de promedios convexos y demostraciones de atención inspiradas en arquitecturas modernas de inteligencia artificial.

El fundamento matemático principal del proyecto proviene del artículo **"The kernel average for two convex functions and its application to the extension and representation of monotone operators"**, de **Heinz H. Bauschke y Xianfu Wang**. En este trabajo, los autores proponen una forma general de combinar dos funciones convexas mediante una función kernel. Esta construcción permite recuperar, como casos particulares, promedios conocidos en análisis convexo, como el promedio aritmético, el promedio epigráfico y el promedio proximal.

El objetivo de KAvgLab no es implementar únicamente una fórmula matemática, sino convertir esa idea en una herramienta computacional reproducible. El software permite definir funciones convexas simples, seleccionar un kernel, evaluar puntos de prueba, comparar diferentes tipos de promedios y exportar resultados en CSV para análisis posterior. Además, incorpora una verificación numérica de la relación entre el promedio kernel y la conjugación de Fenchel, lo que permite estudiar de forma experimental la correspondencia entre el problema primal y su contraparte dual.

Desde el punto de vista de ingeniería, el proyecto separa claramente la configuración, las funciones convexas, los kernels, los solvers, la verificación de Fenchel, la exportación de resultados y las demostración de atención. Esta separación facilita que el software sea inspeccionable, extensible.

La versión actual también conecta el análisis convexo con inteligencia artificial. En particular, implementa una demostración de atención regularizada donde la distribución de atención no depende solo de los scores, sino también de una penalización hacia una distribución previa. Esta idea permite interpretar la atención como un problema de optimización regularizada, lo cual abre una conexión natural con Transformers, LLMs, MLLMs y sistemas de agentes.

En el contexto de Transformers, el software permite comparar la atención softmax estándar con una atención regularizada por kernel. En el contexto de LLMs, incorpora máscaras causales que impiden atender a posiciones futuras, imitando el principio de atención autoregresiva. 

En el contexto de MLLMs, incluye una demostración de cross-attention multimodal sintética, donde consultas de texto pueden atender a claves y valores asociados a regiones visuales simuladas. Finalmente, en el contexto de agentes, incluye un barrido experimental que prueba combinaciones de hiperparámetros y ordena los resultados según objetivos definidos.

KAvgLab debe entenderse como un prototipo mayor: no implementa un Transformer completo ni entrena un LLM, pero sí proporciona una base matemática y computacional para estudiar cómo ideas de análisis convexo, regularización y promedios kernel pueden relacionarse con mecanismos modernos de atención.


Convención del proyecto:

- Comentarios, README y mensajes de consola en español.
- Funciones, structs, enums, módulos y métodos en inglés.
- Resultados exportables en CSV para análisis y presentación.



#### 1. Qué hace el software

El binario `kavg-lab` permite ejecutar seis flujos principales:

| Comando | Qué hace | Relación conceptual |
|---|---|---|
| `compute` | Calcula un kernel average en puntos definidos por YAML | Análisis convexo / optimización |
| `compare` | Compara promedio aritmético, epigráfico y proximal/kernel average | Comparación de promedios convexos |
| `verify-fenchel` | Verifica numéricamente una identidad de Fenchel | Dualidad convexa |
| `attention-demo` | Compara softmax attention vs atención regularizada por kernel | Transformers |
| `multihead-attention-demo` | Ejecuta varias cabeceras de atención con priors distintos | Multi-head attention |
| `agent-sweep` | Prueba hiperparámetros y rankea configuraciones | Agente experimental reproducible |


#### 2. Instalación de Rust en Linux Ubuntu

Rust se instala con `rustup`, que incluye `rustc`, `cargo`, `rustfmt` y herramientas del ecosistema.

##### 2.1 Instalar dependencias básicas

```bash
sudo apt update
sudo apt install -y curl build-essential pkg-config
```

##### 2.2 Instalar Rust con rustup

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Durante la instalación, elegir la opción por defecto.

##### 2.3 Activar Cargo en la terminal actual

```bash
source "$HOME/.cargo/env"
```

Si no funciona, cerrar y abrir la terminal.

##### 2.4 Verificar instalación

```bash
rustc --version
cargo --version
rustup --version
```


#### 3. Instalación de Rust en Windows

##### 3.1 Instalar con rustup-init

1. Descargar `rustup-init.exe` desde la página oficial de Rust.
2. Ejecutar el instalador.
3. Aceptar la instalación por defecto.
4. Si el instalador lo solicita, instalar **Visual Studio C++ Build Tools**.

### 3.2 Verificar en PowerShell

Abrir una nueva ventana de PowerShell y ejecutar:

```powershell
rustc --version
cargo --version
rustup --version
```

Si `cargo` no se reconoce, cerrar y volver a abrir PowerShell. Rust instala las herramientas normalmente en:

```text
%USERPROFILE%\.cargo\bin
```


#### 4. Abrir el proyecto

##### Linux, macOS o WSL

```bash
cd kavg-lab-5
```

### Windows PowerShell

```powershell
cd .\kavg-lab-5
```

Verificar que exista el archivo principal de Cargo:

```bash
ls Cargo.toml
```

En PowerShell:

```powershell
dir Cargo.toml
```


#### 5. Cargo.lock para reproducibilidad

Para una aplicación CLI, conviene versionar `Cargo.lock`, porque fija las versiones exactas de dependencias transitivas usadas por Cargo.

##### 5.1 Quitar `Cargo.lock` de `.gitignore`

En Linux, macOS, WSL o Git Bash:

```bash
sed -i '/^Cargo.lock$/d' .gitignore
```

En Windows PowerShell:

```powershell
(Get-Content .gitignore) | Where-Object { $_ -ne "Cargo.lock" } | Set-Content .gitignore
```

##### 5.2 Generar Cargo.lock

```bash
cargo generate-lockfile
```

##### 5.3 Verificar que existe

```bash
ls Cargo.lock
```

En PowerShell:

```powershell
dir Cargo.lock
```

##### 5.4 Modo recomendado para Git

```bash
git add Cargo.lock .gitignore
git commit -m "Agregar Cargo.lock para builds reproducibles"
```


#### 6. Validación obligatoria antes de presentar

Ejecutar desde la raíz del proyecto:

```bash
cargo fmt
cargo check
cargo test
cargo clippy -- -D warnings
```

También es recomendable generar el binario optimizado:

```bash
cargo build --release
```

Si todo pasa sin errores, el proyecto está listo para una presentación técnica.

Comando alternativo para verificar formato sin modificar archivos:

```bash
cargo fmt -- --check
```

#### 7. Ecuación del kernel average paso a paso

El módulo convexo trabaja con dos funciones convexas `f1` y `f2`, un peso `lambda1` y un kernel `g`.

##### Paso 1: pesos convexos

Se define:

```text
lambda1 in (0, 1)
lambda2 = 1 - lambda1
```

##### Paso 2: restricción de mezcla

Para calcular el promedio en un punto `x`, se buscan dos puntos auxiliares `y1` y `y2` tales que:

```text
lambda1 * y1 + lambda2 * y2 = x
```

En el código se elimina `y2` usando:

```text
y2 = (x - lambda1 * y1) / lambda2
```

Así el problema se reduce a optimizar únicamente sobre `y1`.

##### Paso 3: kernel cuadrático

El software usa el kernel:

```text
g(z) = 1/2 ||z||²
```

con:

```text
z = y1 - y2
```

##### Paso 4: objetivo del kernel average

El promedio kernel se calcula como:

```text
P(x) = min_{y1,y2} lambda1*f1(y1)
                    + lambda2*f2(y2)
                    + c*lambda1*lambda2*g(y1-y2)

sujeto a: lambda1*y1 + lambda2*y2 = x
```

El factor `c` cambia el tipo de promedio:

```text
c = 0  -> promedio epigráfico
c = 1  -> proximal/kernel average con squared-norm
```

##### Paso 5: promedio aritmético usado para comparar

```text
A(x) = lambda1*f1(x) + lambda2*f2(x)
```

El comando `compare` imprime y exporta:

```text
A(x), E(x), P(x), P(x)-E(x), A(x)-P(x)
```

donde `E(x)` es el promedio epigráfico y `P(x)` es el proximal/kernel average.


#### 8. Identidad de Fenchel verificada

El comando `verify-fenchel` verifica numéricamente una identidad de dualidad:

```text
(P(f1, f2, g))* (s) ≈ P(f1*, f2*, g*) (s)
```

El lado izquierdo se aproxima por maximización:

```text
sup_x <s, x> - P(f1, f2, g)(x)
```

El lado derecho se calcula usando los conjugados analíticos disponibles:

```text
f1*, f2*, g*
```

Para el kernel cuadrático:

```text
g(z) = 1/2 ||z||²
g*(s) = 1/2 ||s||²
```

El CSV exporta el valor izquierdo aproximado, el valor derecho, error absoluto, error relativo y estado `passed`.


#### 9. Ecuación de atención paso a paso

La demo de atención representa `queries`, `keys` y `values` mediante vectores pequeños definidos en YAML.

##### Paso 1: scores de atención

Para una query `q` y keys `k_i`:

```text
score_i = <q, k_i> / sqrt(d)
```

donde `d` es la dimensión de los embeddings.

##### Paso 2: máscara opcional

Si existe máscara causal o custom, se modifica el score:

```text
masked_score_i = score_i + mask_i
```

Las posiciones bloqueadas usan:

```text
-inf
```

Por tanto reciben peso cero.

##### Paso 3: softmax estándar

```text
p_softmax_i = exp(masked_score_i / temperature)
              / sum_j exp(masked_score_j / temperature)
```

##### Paso 4: prior estructural

La atención regularizada usa una distribución previa `p0`.

Si no se define `prior`, el software usa una distribución uniforme sobre los tokens permitidos:

```text
p0_i = 1 / n
```

Si hay máscara, el prior se renormaliza sobre posiciones permitidas.

##### Paso 5: atención regularizada por kernel

El problema resuelto es:

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
- `temperature * sum p_i log(p_i)` produce suavidad entrópica.
- `gamma/2 * ||p-p0||²` acerca la atención al prior estructural.
- La restricción `sum p_i = 1` mantiene una distribución de probabilidad.

##### Paso 6: salida de atención

```text
output = sum_i p_i * value_i
```

El software calcula la salida para softmax estándar y para atención regularizada.


#### 10. Multi-head attention

El comando multi-head ejecuta varias cabeceras con diferentes parámetros:

```text
head_h = Attention(q, K, V, temperature_h, gamma_h, prior_h)
```

Luego agrega sus salidas mediante promedio simple:

```text
aggregated_output = (1/H) * sum_h output_h
```

También calcula diversidad entre cabeceras:

```text
mean_pairwise_l1
mean_pairwise_l2
mean_pairwise_js
```

Esto permite explicar cómo distintas cabeceras atienden a patrones diferentes.


#### 11. Cross-attention multimodal sintética

El ejemplo multimodal usa alias semánticos:

```yaml
text_queries: [...]
image_keys: [...]
image_values: [...]
```

Internamente equivalen a:

```yaml
queries: [...]
keys: [...]
values: [...]
```

La interpretación es:

```text
texto consulta regiones visuales sintéticas
```

Esto permite presentar una idea tipo MLLM sin depender todavía de un encoder real de imágenes.


#### 12. Agent sweep

El comando `agent-sweep` prueba combinaciones de:

```text
gamma
temperature
prior
```

Para cada configuración ejecuta la demo de atención, calcula métricas promedio y asigna un score según el objetivo.

Objetivos disponibles:

```text
max-entropy
min-distance-to-prior
max-difference-from-softmax
min-output-shift
balanced-tradeoff
```

Para `balanced-tradeoff`, el score usado es:

```text
score = entropy
        + 0.5 * js_softmax_regularized
        - 0.5 * distance_to_prior
        - 0.25 * output_shift
```

Interpretación: busca una atención expresiva, no demasiado alejada del prior y sin desplazar excesivamente la salida.


#### 13. Comandos para ejecutar y explicar el software

##### 13.1 Ver ayuda general

```bash
cargo run -- --help
```

##### 13.2 Ver ayuda de un subcomando

```bash
cargo run -- compute --help
cargo run -- compare --help
cargo run -- verify-fenchel --help
cargo run -- attention-demo --help
cargo run -- multihead-attention-demo --help
cargo run -- agent-sweep --help
```

#### 14. Demostración de análisis convexo

##### 14.1 Cálculo

```bash
cargo run -- compute \
  --config examples/quadratic_l1.yaml \
  --output results.csv
```

Explicación breve:

```text
Calcula P(x) para varios puntos x usando f1 cuadrática, f2 L1 y kernel squared-norm.
```

##### 14.2 Comparación

```bash
cargo run -- compare \
  --config examples/compare_quadratic_l1.yaml \
  --output comparison.csv
```

Explicación breve:

```text
Compara promedio aritmético, promedio epigráfico y proximal/kernel average.
```

##### 14.3 Verificación  Fenchel

```bash
cargo run -- verify-fenchel \
  --config examples/fenchel_quadratic_l2.yaml \
  --output fenchel.csv
```

Explicación breve:

```text
Verifica numéricamente una identidad primal-dual usando conjugados convexos.
```


#### 15. Demostración con OSQP

```bash
cargo run -- compute \
  --config examples/quadratic_l1_osqp.yaml \
  --output results_osqp.csv
```

```bash
cargo run -- compare \
  --config examples/compare_quadratic_l1_osqp.yaml \
  --output comparison_osqp.csv
```

```bash
cargo run -- verify-fenchel \
  --config examples/fenchel_quadratic_l2_osqp.yaml \
  --output fenchel_osqp.csv
```

Explicación breve:

```text
Usa OSQP como backend alternativo cuando el problema puede expresarse como QP.
```


#### 16. Demos de atención, Transformers, LLM y MLLM

##### 16.1 Attention demostración base

```bash
cargo run -- attention-demo \
  --config examples/attention_demo.yaml \
  --output attention_results.csv
```

Explicación breve:

```text
Compara softmax attention contra atención regularizada por kernel.
```

Métricas exportadas:

```text
weight_l1_distance
weight_l2_distance
output_l2_distance
standard_entropy
regularized_entropy
kl_regularized_to_softmax
kl_regularized_to_prior
js_softmax_regularized
effective_tokens_standard
effective_tokens_regularized
standard_top1_mass
regularized_top1_mass
standard_topk_mass
regularized_topk_mass
```

##### 16.2 Attention con prior uniforme

```bash
cargo run -- attention-demo \
  --config examples/attention_demo_uniform.yaml \
  --output attention_uniform.csv
```

##### 16.3 Causal mask tipo LLM

```bash
cargo run -- attention-demo \
  --config examples/attention_causal.yaml \
  --output attention_causal.csv
```

Explicación breve:

```text
Bloquea tokens futuros. Es la conexión más directa con atención autoregresiva tipo LLM.
```

##### 16.4 Máscara personalizada

```bash
cargo run -- attention-demo \
  --config examples/attention_custom_mask.yaml \
  --output attention_custom_mask.csv
```

Ejemplo de máscara custom:

```yaml
mask:
  type: custom
  matrix:
    - [0, "-inf", "-inf"]
    - [0, 0, "-inf"]
    - [0, 0, 0]
```

##### 16.5 Multi-head attention

```bash
cargo run -- multihead-attention-demo \
  --config examples/multihead_attention.yaml \
  --output multihead_results.csv
```

Explicación breve:

```text
Ejecuta varias cabeceras con distintos priors, gamma y temperatura.
```

##### 16.6 Cross-attention multimodal sintética

```bash
cargo run -- attention-demo \
  --config examples/cross_attention_multimodal.yaml \
  --output cross_attention.csv
```

Explicación breve:

```text
Simula texto atendiendo a regiones visuales representadas por embeddings pequeños.
```

##### 16.7 Agent sweep

```bash
cargo run -- agent-sweep \
  --config examples/attention_sweep.yaml \
  --output attention_sweep.csv
```

Explicación breve:

```text
Prueba configuraciones de atención y devuelve un ranking reproducible.
```


#### 17. Resultados de ejemplo para visualizar

La carpeta `sample_outputs/` incluye vistas previas en CSV para enseñar capturas o explicar el formato de salida.

> Importante: estos CSV son una vista previa orientativa. Para resultados definitivos, regenerarlos ejecutando los comandos con `cargo run` en tu máquina.

Archivos incluidos:

```text
sample_outputs/results_preview.csv
sample_outputs/comparison_preview.csv
sample_outputs/attention_results_preview.csv
sample_outputs/attention_causal_preview.csv
sample_outputs/cross_attention_preview.csv
sample_outputs/multihead_results_preview.csv
sample_outputs/attention_sweep_preview.csv
```

##### 17.1 Vista previa: `compute`

Comando que genera el resultado definitivo:

```bash
cargo run -- compute --config examples/quadratic_l1.yaml --output results.csv
```

Extracto visual (ejemplo):

```text
index | point                    | value        | y1                       | y2
0     | [1.0000000000,1.0000000000] | 1.1850000143 | [0.8999023438,0.8999023438] | [1.1000976562,1.1000976562]
1     | [2.0000000000,-1.0000000000] | 2.1516666740 | [1.5666503906,-0.8999023438] | [2.4333496094,-1.1000976562]
2     | [0.0000000000,0.5000000000] | 0.2341666669 | [0.0000000000,0.5666503906] | [0.0000000000,0.4333496094]
```

Lectura rápida:

```text
El solver busca y1,y2 que mezclan al punto x y reducen el objetivo convexo con penalización kernel.
```

##### 17.2 Vista previa: `compare`

Comando que genera el resultado definitivo:

```bash
cargo run -- compare --config examples/compare_quadratic_l1.yaml --output comparison.csv
```

Extracto visual (ejemplo):

```text
index | arithmetic | epigraphical | proximal   | arithmetic - proximal
0     | 1.200000   | 1.155000     | 1.185000   | 0.015000
1     | 2.300000   | 1.855000     | 2.151667   | 0.148333
2     | 0.237500   | 0.227500     | 0.234167   | 0.003333
3     | 2.300000   | 1.855000     | 2.151667   | 0.148333
```

Lectura rápida:

```text
El promedio proximal queda entre el epigráfico y el aritmético en estos ejemplos.
```

##### 17.3 Vista previa: `attention-demo`

Comando que genera el resultado definitivo:

```bash
cargo run -- attention-demo --config examples/attention_demo.yaml --output attention_results.csv
```

Extracto visual (ejemplo):

```text
query | softmax weights                          | regularized weights                      | JS distance | effective tokens reg
0     | [0.3463,0.1871,0.1604,0.3062]              | [0.2914,0.1920,0.1715,0.3451]            | 0.0433      | 3.8409
1     | [0.1604,0.3464,0.2183,0.2750]              | [0.1714,0.2912,0.2141,0.3234]            | 0.0477      | 3.8830
2     | [0.2195,0.1882,0.3762,0.2161]              | [0.2167,0.1943,0.3088,0.2803]            | 0.0614      | 3.9315
```

Lectura rápida:

```text
La regularización mueve la atención hacia el prior [0.20,0.20,0.20,0.40], aumentando la masa del cuarto token.
```

##### 17.4 Vista previa: máscara causal tipo LLM

Comando que genera el resultado definitivo:

```bash
cargo run -- attention-demo --config examples/attention_causal.yaml --output attention_causal.csv
```

Extracto visual  (ejemplo):

```text
query | masked_scores                          | regularized_weights
0     | [0.5773502692,-inf,-inf,-inf]             | [1.0000,0.0000,0.0000,0.0000]
1     | [0.2309401077,0.5773502692,-inf,-inf]     | [0.4424,0.5576,0.0000,0.0000]
2     | [0.1154700538,0.1732050808,0.5773502692,-inf] | [0.2894,0.3026,0.4080,0.0000]
```

Lectura rápida:

```text
La query 0 solo puede mirar el token 0; la query 1 puede mirar 0 y 1; la query 2 puede mirar 0,1,2.
```

##### 17.5 Vista previa: cross-attention multimodal

Comando que genera el resultado definitivo:

```bash
cargo run -- attention-demo --config examples/cross_attention_multimodal.yaml --output cross_attention.csv
```

Extracto visual (ejemplo):

```text
query | regularized_weights                      | regularized_output
0     | [0.2825,0.2238,0.1973,0.2964]              | [0.4307,0.3720,0.3455]
1     | [0.1916,0.2064,0.2920,0.3099]              | [0.3466,0.3614,0.4470]
```

Lectura rápida:

```text
Los tokens textuales atienden a regiones visuales sintéticas; el prior favorece la región global.
```

##### 17.6 Vista previa: multi-head attention

Comando que genera el resultado definitivo:

```bash
cargo run -- multihead-attention-demo --config examples/multihead_attention.yaml --output multihead_results.csv
```

Extracto visual (ejemplo):

```text
query | aggregated_output     | average_entropy | mean_pairwise_js
0     | [0.6371,0.5446]      | 1.3605          | 0.0572
1     | [0.5352,0.6363]      | 1.3688          | 0.0485
```

Lectura rápida:

```text
Cada cabecera produce una distribución distinta; el CSV resume salida agregada y diversidad entre cabeceras.
```

##### 17.7 Vista previa: agent sweep

Comando que genera el resultado definitivo:

```bash
cargo run -- agent-sweep --config examples/attention_sweep.yaml --output attention_sweep.csv
```

Extracto visual (ejemplo):

```text
rank | gamma | temperature | prior_name | score   | mean_effective_tokens
1    | 2.0   | 1.5         | uniform    | 1.3297  | 3.9742
2    | 1.0   | 1.5         | uniform    | 1.3192  | 3.9665
3    | 0.5   | 1.5         | uniform    | 1.3126  | 3.9612
4    | 2.0   | 1.0         | uniform    | 1.3099  | 3.9544
```

Lectura rápida:

```text
El agente experimental ordena configuraciones según el objetivo balanced-tradeoff.
```


> La principal contribución del MVP es tender un puente entre teoría convexa y mecanismos de inteligencia artificial. La atención regularizada 
> implementada en el software permite reinterpretar una parte central de los Transformers como un problema de optimización: los pesos de atención no se calculan únicamente a partir de compatibilidades entre vectores, sino que también pueden incorporar priors, restricciones, máscaras y regularización.

Esta perspectiva es relevante porque muchos sistemas modernos de IA combinan aprendizaje estadístico con restricciones estructurales. En LLMs, las máscaras causales imponen una estructura temporal, en MLLMs, la cross-attention conecta modalidades distintas y en agentes, la selección de configuraciones puede verse como un proceso de búsqueda guiada por métricas. KAvgLab no pretende reemplazar frameworks de deep learning, sino ofrecer un entorno pequeño, transparente y matemáticamente interpretable para estudiar estas ideas desde la optimización convexa.
