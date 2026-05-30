# Política de seguridad

KAvgLab es un CLI experimental y académico. No está diseñado para procesar secretos, credenciales, datos personales sensibles ni entradas no confiables en entornos multiusuario.

## Versiones soportadas

| Versión | Soporte |
|---|---|
| `main` | Activo |
| Releases anteriores | Mejor esfuerzo |

## Reportar problemas

Para reportar un problema de seguridad, abrir un issue privado si el repositorio lo permite o contactar al mantenedor del proyecto. No publiques credenciales, tokens, llaves privadas ni datos sensibles en issues públicos.

## Alcance actual

Son relevantes para este proyecto:

- Lectura segura de archivos YAML locales.
- Manejo claro de errores al parsear configuraciones.
- Evitar escrituras fuera de las rutas indicadas por el usuario.
- Evitar pánicos en entradas inválidas cuando sea posible devolver un error controlado.
- Mantener dependencias mínimas y auditables.

Fuera de alcance por ahora:

- Ejecución remota.
- Servicio web persistente.
- Autenticación de usuarios.
- Almacenamiento de secretos.
