# Dataset Tarea 2 para anotacion y modelado


## Contenido

- `tarea2_dataset_candidatos_2000.jsonl`: corpus completo curado.
- `tarea2_train_weak.jsonl`: entrenamiento inicial con etiquetas heuristicas.
- `tarea2_val_weak.jsonl`: validacion inicial con etiquetas heuristicas.
- `tarea2_gold_eval_pendiente.jsonl`: 10% reservado para anotacion manual.
- `tarea2_muestra_anotacion_10pct.csv`: misma muestra en CSV para anotar.
- `tarea2_estadisticas_modelado.json`: diagnostico de la seleccion.
- `GUIA_ANOTACION_TAREA2.md`: criterios breves para anotadores.

## Uso recomendado

1. Anotar `tarea2_muestra_anotacion_10pct.csv`.
2. No usar `gold_eval_pendiente` para entrenar hasta cerrar la anotacion.
3. Entrenar una primera linea base con `tarea2_train_weak.jsonl`.
4. Ajustar umbrales o reglas usando `tarea2_val_weak.jsonl`.
5. Evaluar contra la muestra anotada manualmente cuando este consolidada.

## Resumen

- Total candidatos: 2000
- Positivos: 1000
- Negativos: 1000
- Muestra manual: 200
- Documentos unicos: 2000
