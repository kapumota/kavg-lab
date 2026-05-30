# Fase 3: solvers, algoritmos y comparación desde CLI

Esta fase mantiene la identidad del proyecto como herramienta de línea de comandos y agrega una capa algorítmica más fuerte. El objetivo no es crear una interfaz gráfica, sino convertir `kavg-lab` en un laboratorio CLI para comparar métodos de optimización aplicados a promedios kernel y atención regularizada.

## Solvers convexos agregados

Además de `coordinate-descent`, `subgradient` y `osqp`, se agregan métodos experimentales:

- `proximal-gradient`
- `fista`
- `admm`

El soporte OSQP sigue siendo parcial y está orientado a casos cuadráticos compatibles. Los nuevos métodos se integran al mismo flujo YAML mediante `solver.method`.

## Solvers para atención sobre simplex

Para atención regularizada, los pesos viven en el simplex:

```text
p_i >= 0
sum_i p_i = 1
```

Por eso esta fase agrega tres métodos orientados a esa geometría:

- `projected-gradient`
- `mirror-descent`
- `frank-wolfe`

`mirror-descent` usa actualizaciones multiplicativas y es natural para distribuciones de probabilidad. `frank-wolfe` produce actualizaciones hacia vértices del simplex, útil para estudiar atención más dispersa.

## Uso desde CLI

Ejecutar atención con mirror descent:

```bash
cargo run -- attention-demo \
  --config examples/attention_demo.yaml \
  --solver mirror-descent \
  --output sample_outputs/attention_mirror.csv
```

Ejecutar atención con Frank-Wolfe:

```bash
cargo run -- attention-demo \
  --config examples/attention_demo.yaml \
  --solver frank-wolfe \
  --output sample_outputs/attention_frank_wolfe.csv
```

Comparar solvers convexos:

```bash
cargo run -- compare-solvers \
  --config examples/quadratic_l1.yaml \
  --solvers coordinate-descent,subgradient,osqp,proximal-gradient,fista \
  --output sample_outputs/solver_comparison.csv
```

## Alcance

Esta fase no afirma que todos los métodos sean solvers industriales completos. Se agregan como implementaciones experimentales y comparables, útiles para enseñanza avanzada, investigación reproducible y análisis de comportamiento algorítmico.

Una fase posterior debería reforzar:

- operadores proximales específicos por función;
- ADMM con separación primal-dual más formal;
- restricciones duras generales para OSQP;
- pruebas de propiedades y benchmarks sistemáticos.
