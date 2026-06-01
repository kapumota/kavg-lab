# Fase 4: experimentación CLI reproducible

La Fase 4 mantiene la identidad del proyecto como herramienta de línea de comandos y agrega un flujo reproducible de ejecución experimental. En lugar de incorporar dashboard, servidor web o visualizaciones, se introduce el comando `run-suite`, que orquesta varios experimentos y genera un paquete de evidencia autocontenido.

## Objetivo

Convertir KAvgLab en un CLI capaz de producir evidencia reproducible para:

- cálculo de promedios kernel;
- verificación de Fenchel;
- atención regularizada sobre simplex;
- comparación de solvers.

La salida queda organizada en un directorio versionable externamente, pero no debe subirse al repositorio como artefacto ordinario.

## Comando principal

```bash
kavg-lab run-suite \
  --suite experiments/suite.yaml \
  --out evidence/run_001
```

## Estructura generada

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

## Archivo `manifest.json`

El manifiesto registra metadatos mínimos de trazabilidad:

- nombre y versión de la herramienta;
- versión de `rustc`;
- tiempo de inicio y fin en milisegundos Unix;
- duración de ejecución;
- ruta de la suite;
- hash FNV-1a de la suite YAML;
- commit Git detectado;
- número de pasos ejecutados;
- estado final.

## Archivo `commands.log`

`commands.log` contiene los comandos CLI equivalentes a los pasos ejecutados por la suite. Esto permite reproducir manualmente cada etapa sin depender de una interfaz gráfica.

## Archivo `summary.json`

`summary.json` resume cada paso ejecutado, su configuración, archivo de salida, número de resultados y estado. Es útil para auditoría automática, comparación entre corridas o integración posterior con CI.

## Decisión de diseño

Esta fase no intenta resolver dashboards, gráficos ni reportes visuales. Su valor está en la reproducibilidad: la misma suite YAML produce la misma estructura de evidencia y deja constancia explícita de versión, commit, comandos y resultados.
