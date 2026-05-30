# Registro de cambios

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
