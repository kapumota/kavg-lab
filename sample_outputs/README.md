# Salidas de ejemplo

Este directorio documenta la convención esperada para salidas generadas por el CLI.

No es obligatorio versionar resultados grandes. Para una demostración pequeña, se recomienda generar:

```bash
kavg-lab compute \
  --config examples/quadratic_l1.yaml \
  --output sample_outputs/results.csv \
  --json sample_outputs/results.json \
  --manifest sample_outputs/manifest.json
```

Si los resultados son temporales o voluminosos, mantenerlos fuera de Git.
