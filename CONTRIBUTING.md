# Contribución

Gracias por contribuir a KAvgLab. Este proyecto mantiene una identidad de CLI científico: los cambios deben mejorar la reproducibilidad, la calidad matemática, la robustez del código o la capacidad experimental sin convertir el proyecto en dashboard o aplicación web.

## Flujo recomendado

1. Crear una rama descriptiva:

```bash
git switch -c nombre-corto-del-cambio
```

2. Ejecutar validaciones locales antes de abrir un Pull Request:

```bash
cargo fmt -- --check
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
cargo build --release
```

3. Mantener los ejemplos YAML pequeños, reproducibles y documentados.

4. No eliminar la salida CSV existente cuando se agreguen nuevos formatos de salida.

## Estilo del proyecto

- Comentarios, documentación y mensajes de consola: español.
- Identificadores de Rust: inglés.
- Cambios matemáticos: acompañar con una explicación breve y al menos una prueba.
- Nuevos comandos CLI: documentar en `README.md` y agregar ejemplo en `examples/` o `experiments/`.

## Tipos de contribución valiosos

- Nuevos kernels convexos.
- Nuevas funciones convexas y conjugados de Fenchel.
- Solvers reproducibles y comparables.
- Pruebas de integración y pruebas de propiedades.
- Exportación de evidencia experimental en formatos CLI.
- Mejoras de rendimiento que no sacrifiquen determinismo.
