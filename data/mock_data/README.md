# Datos Mock — Tarea 1 IMRaD

Este directorio contiene datos sintéticos que simulan el corpus de artículos
científicos en español utilizado durante la fase de desarrollo del modelo.

## Justificación

El pipeline de datos real (descarga de PDFs, OCR, anotación manual, cálculo
de acuerdo interanotador) se ejecuta en paralelo con el desarrollo del modelo.
Para evitar bloqueos por dependencias, se generaron estos datos mock que
reproducen la estructura y distribución esperadas del corpus final.

## Contenido

| Carpeta | Descripción |
|---------|-------------|
| `processed/` | Dataset completo en JSONL (200 ejemplos, 50 por etiqueta) |
| `splits/` | DatasetDict con splits train(140)/val(30)/test(30) estratificados |
| `annotated/` | Muestra de 10 ejemplos para validación del proceso de anotación |

## Formato

Cada ejemplo contiene:
- `id`: identificador único
- `texto`: fragmento en español
- `etiqueta`: índice IMRaD (0=Introducción, 1=Metodología, 2=Resultados, 3=Discusión)
- `nombre_etiqueta`: nombre de la categoría

## Reemplazo

Al completar el pipeline real, reemplazar este directorio con los datos
anotados y validados por humanos (con Fleiss' κ ≥ 0.6).
