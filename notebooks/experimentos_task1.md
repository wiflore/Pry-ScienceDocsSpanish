# Experimentos — Tarea 1: Clasificación Retórica (8 clases)

**Proyecto:** Análisis de documentos científicos en español  
**Grupo:** FLAG – TICsW, Universidad de los Andes  
**Última actualización:** 2026-05-06 (Qwen3 few-shot completado)

---

## Configuración experimental común

### Dataset
| Parámetro | Valor |
|---|---|
| Fuente | `data/RawDatasetsV2/DatasetAnotacionManual_Consolidado.xlsx` |
| Total muestras (tras limpiar NaN) | 1 587 |
| Etiqueta usada | `etiqueta_anotador` (validación humana, NO `etiqueta_original`) |
| Documentos únicos | 936 |
| Split | 80 / 10 / 10 por documento (seed=42, sin solapamiento) |
| Train / Val / Test | 1 268 / 158 / 161 |

### Distribución de clases (dataset completo)
| Clase | N total | N test | Descripción |
|---|---|---|---|
| BACK | 413 | 52 | Antecedentes / Estado del arte |
| DISC | 264 | 27 | Discusión |
| METH | 243 | 28 | Metodología |
| INTRO | 202 | 17 | Introducción |
| RES | 196 | 14 | Resultados |
| CONC | 110 | 11 | Conclusiones |
| LIM | 97 | 9 | Limitaciones |
| CONTR | 62 | 3 | Contribuciones |

> **Nota:** La clase CONTR tiene solo 3 muestras en test. Su F1 es poco fiable estadísticamente y debe interpretarse con cautela.

### Métrica principal
**Macro F1** (promedio no ponderado sobre las 8 clases). Se reportan también Accuracy, Precision y Recall por clase.

---

## Experimento 1 — Baseline: TF-IDF + Regresión Logística

**Script:** `scripts/train_task1_tfidf.py`  
**Fecha:** 2026-05-06  
**Estado:** ✅ Completado

### Configuración
| Parámetro | Valor |
|---|---|
| Vectorizador | TF-IDF, ngram=(1,2), max_features=100 000, sublinear_tf=True, min_df=2 |
| Clasificador | Logistic Regression, C=1.0, max_iter=1000, solver=lbfgs, multinomial |
| Pesos de clase | Inverso de frecuencia (class_weight=dict) |

### Resultados en test
| Métrica | Valor |
|---|---|
| **Macro F1** | **0.2993** |
| Accuracy | 0.4348 |

#### F1 por clase
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BACK | 0.50 | 0.52 | 0.51 | 52 |
| CONC | 0.50 | 0.09 | 0.15 | 11 |
| CONTR | 0.00 | 0.00 | 0.00 | 3 |
| DISC | 0.44 | 0.56 | 0.49 | 27 |
| INTRO | 0.22 | 0.29 | 0.25 | 17 |
| LIM | 0.25 | 0.11 | 0.15 | 9 |
| METH | 0.50 | 0.64 | 0.56 | 28 |
| RES | 0.38 | 0.21 | 0.27 | 14 |

#### Matriz de confusión
```
           BACK  CONC  CONTR  DISC  INTRO  LIM  METH  RES
BACK         27     1      0    10      6    1     6    1
CONC          4     1      0     2      3    1     0    0
CONTR         2     0      0     0      1    0     0    0
DISC          7     0      0    15      1    1     2    1
INTRO         7     0      0     3      5    0     1    1
LIM           4     0      0     0      1    1     3    0
METH          2     0      0     3      3    0    18    2
RES           1     0      0     1      3    0     6    3
```

### Observaciones
- Sirve como **piso de rendimiento**: cualquier modelo más complejo debe superar 0.30 Macro F1.
- METH es la clase más fácil para TF-IDF (0.56 F1) por su vocabulario técnico específico.
- CONTR y LIM obtienen F1=0.00 y 0.15: vocabulario demasiado solapado con otras clases.
- Los errores más frecuentes: BACK↔DISC, INTRO↔BACK, RES→METH.

---

## Experimento 2 — SciBETO-large fine-tuned (v2, early stopping)

**Script:** `scripts/train_task1_scibeto.py`  
**Fecha:** 2026-05-06  
**Estado:** ✅ Completado  
**Modelo base:** `Flaglab/SciBETO-large` (RoBERTa-large, 355M parámetros, preentrenado en 11B tokens de texto científico en español)

### Configuración
| Parámetro | Valor |
|---|---|
| Tokenización | Head+Tail: 128 tokens cabeza + 382 tokens cola + 2 especiales = 512 |
| Batch size | 8 |
| Learning rate | 2e-5 (AdamW) |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Classifier dropout | 0.3 |
| Hidden dropout | 0.3 |
| Función de pérdida | CrossEntropyLoss con pesos por clase (inverso de frecuencia) |
| Épocas máximas | 10 |
| Early stopping patience | 3 (criterio: val Macro F1) |

#### Pesos de clase usados
| Clase | Peso |
|---|---|
| BACK | 0.505 |
| DISC | 0.741 |
| METH | 0.839 |
| RES | 0.991 |
| INTRO | 0.943 |
| CONC | 1.865 |
| LIM | 1.887 |
| CONTR | 2.935 |

### Curva de entrenamiento
| Época | Train Loss | Val Loss | Val Macro F1 | Mejor |
|---|---|---|---|---|
| 1 | 2.0881 | 1.7868 | 0.2847 | ✓ |
| 2 | 1.6674 | 1.4323 | 0.4327 | ✓ |
| 3 | 1.2470 | 1.3952 | **0.5435** | ✓ |
| 4 | 0.9542 | 1.4837 | 0.4821 | – (1/3) |
| 5 | 0.6395 | 1.7780 | 0.4536 | – (2/3) |
| 6 | 0.4335 | 1.8874 | 0.5077 | – (3/3) → **Early stop** |

> Mejor checkpoint: época 3 (val Macro F1 = 0.5435). El modelo se guarda en `models/scibeto-task1-8clases/best_model/`.

### Resultados en test (checkpoint época 3)
| Métrica | Valor |
|---|---|
| **Macro F1** | **0.3983** |
| Accuracy | 0.4472 |

#### F1 por clase
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BACK | 0.55 | 0.33 | 0.41 | 52 |
| CONC | 0.33 | 0.27 | 0.30 | 11 |
| CONTR | 0.17 | 0.33 | 0.22 | 3 |
| DISC | 0.44 | 0.56 | 0.49 | 27 |
| INTRO | 0.32 | 0.47 | 0.38 | 17 |
| LIM | 0.26 | 0.56 | 0.36 | 9 |
| METH | 0.76 | 0.68 | 0.72 | 28 |
| RES | 0.33 | 0.29 | 0.31 | 14 |

### Observaciones
- **+10pp sobre TF-IDF** (0.40 vs 0.30). La representación contextual de SciBETO captura mejor la semántica retórica.
- METH sigue siendo la clase más fácil (0.72), probablemente por vocabulario técnico diferenciado.
- CONTR/LIM mejoran respecto a TF-IDF pero siguen siendo las clases más difíciles.
- BACK tiene recall bajo (0.33): el modelo confunde muchos fragmentos de BACK con DISC e INTRO, lo que es razonable semánticamente.
- El sobreajuste es visible: train_loss cae a 0.43 mientras val_loss sube a 1.89. Early stopping evitó que empeorara más.
- **Brecha val_f1 (0.54) vs test_f1 (0.40)**: el conjunto de validación puede no ser representativo de la distribución de test. Con solo 158 muestras de val, la estimación es ruidosa.

---

## Experimento 3 — Gemini 2.5 Flash (zero-shot y few-shot)

**Script:** `scripts/eval_task1_gemini.py`  
**Fecha:** 2026-05-06  
**Estado:** ⏳ En ejecución  
**Modelo:** `gemini-2.5-flash` (Google, consumido vía API)

### Configuración
| Parámetro | Valor |
|---|---|
| Temperatura | 0.0 |
| Max output tokens | 10 |
| Thinking budget | 0 (desactivado) |
| Rate limit | ~14 req/min (4.2s entre llamadas) |
| Muestras evaluadas | 161 (mismo test set) |

#### Prompt zero-shot
- Sistema: definiciones de las 8 clases + 8 reglas de desambiguación en orden de prioridad
- Usuario: `Fragmento: "{text}"\nEtiqueta:`

#### Prompt few-shot
- Igual que zero-shot + 1 ejemplo por clase (8 ejemplos) sampleados del train set (seed=42)
- Los ejemplos se truncan a 400 palabras

### Resultados en test
| Métrica | Zero-shot | Few-shot |
|---|---|---|
| **Macro F1** | **0.4730** | **0.4878** |
| Accuracy | 0.6211 | 0.6335 |
| Latencia media/muestra | 0.96s | 0.84s |
| Tiempo total | 155s (~2.6 min) | 135s (~2.3 min) |
| Respuestas no parseadas | 0/161 | 0/161 |

#### F1 por clase — Zero-shot
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BACK | 0.56 | 0.79 | 0.66 | 52 |
| CONC | 0.80 | 0.36 | 0.50 | 11 |
| CONTR | 0.00 | 0.00 | 0.00 | 3 |
| DISC | 0.80 | 0.59 | 0.68 | 27 |
| INTRO | 0.35 | 0.41 | 0.38 | 17 |
| LIM | 0.40 | 0.22 | 0.29 | 9 |
| METH | 0.76 | 0.93 | 0.84 | 28 |
| RES | 1.00 | 0.29 | 0.44 | 14 |

#### F1 por clase — Few-shot (1 ejemplo/clase)
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BACK | 0.63 | 0.75 | 0.68 | 52 |
| CONC | 0.60 | 0.55 | 0.57 | 11 |
| CONTR | 0.00 | 0.00 | 0.00 | 3 |
| DISC | 0.62 | 0.67 | 0.64 | 27 |
| INTRO | 0.37 | 0.41 | 0.39 | 17 |
| LIM | 0.67 | 0.22 | 0.33 | 9 |
| METH | 0.84 | 0.93 | 0.88 | 28 |
| RES | 0.67 | 0.29 | 0.40 | 14 |

### Observaciones
- Gemini **sin fine-tuning supera a SciBETO fine-tuneado** (+7.5pp Macro F1). La capacidad de comprensión semántica del LLM compensa la falta de entrenamiento específico.
- Few-shot mejora respecto a zero-shot (+1.5pp) pero la ganancia es modesta — los 8 ejemplos ayudan principalmente en CONC (+7pp F1) y DISC (-4pp, ligera regresión).
- METH sigue siendo la clase más fácil (0.84–0.88) para todos los modelos.
- RES tiene precision perfecta en zero-shot (1.00) pero recall muy bajo (0.29): el modelo es conservador, solo predice RES cuando está muy seguro.
- CONTR = 0.00 F1 en ambos modos. Con solo 3 muestras en test, no es conclusivo.
- 0 respuestas no parseadas: Gemini siempre devuelve una etiqueta válida.

---

## Experimento 4 — Qwen3-8B local (HuggingFace Transformers, zero-shot y few-shot)

**Script:** `scripts/eval_task1_qwen.py`  
**Fecha:** 2026-05-06  
**Estado:** ✅ Completado  
**Modelo:** `Qwen/Qwen3-8B` (8B parámetros, inferencia local vía HuggingFace `transformers`)  
**Hardware:** NVIDIA RTX 4070, 12.9 GB VRAM, CUDA 11.8

### Configuración
| Parámetro | Valor |
|---|---|
| Dtype | `torch.bfloat16` |
| device_map | `auto` (CUDA) |
| Thinking mode | Desactivado (`enable_thinking=False`) |
| Decodificación | Greedy (`do_sample=False`) |
| Max new tokens | 15 |
| Temperatura | – (greedy) |
| Few-shot ejemplos | 1 por clase (8 total), truncados a 80 palabras por ejemplo |

> **Nota:** Se usa HuggingFace `transformers` (no Ollama) para compatibilidad con despliegue en AWS.

### Resultados en test
| Métrica | Zero-shot | Few-shot |
|---|---|---|
| **Macro F1** | **0.4165** | **0.3538** |
| Accuracy | 0.5217 | 0.4534 |
| Latencia media/muestra | 2.81s | 11.12s |
| Tiempo total | ~452s (~7.5 min) | ~1791s (~30 min) |
| Respuestas no parseadas | 0/161 | 0/161 |

#### F1 por clase — Zero-shot
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BACK | 0.64 | 0.56 | 0.60 | 52 |
| CONC | 0.33 | 0.36 | 0.35 | 11 |
| CONTR | 0.00 | 0.00 | 0.00 | 3 |
| DISC | 0.37 | 0.37 | 0.37 | 27 |
| INTRO | 0.32 | 0.59 | 0.42 | 17 |
| LIM | 1.00 | 0.22 | 0.36 | 9 |
| METH | 0.79 | 0.82 | 0.81 | 28 |
| RES | 0.43 | 0.43 | 0.43 | 14 |

#### F1 por clase — Few-shot (1 ejemplo/clase, 80 palabras)
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| BACK | 0.59 | 0.31 | 0.41 | 52 |
| CONC | 0.50 | 0.27 | 0.35 | 11 |
| CONTR | 0.00 | 0.00 | 0.00 | 3 |
| DISC | 0.28 | 0.81 | 0.41 | 27 |
| INTRO | 0.54 | 0.41 | 0.47 | 17 |
| LIM | 1.00 | 0.11 | 0.20 | 9 |
| METH | 0.76 | 0.79 | 0.77 | 28 |
| RES | 0.50 | 0.14 | 0.22 | 14 |

### Observaciones
- **Few-shot empeora respecto a zero-shot** (−6.3pp Macro F1). Fenómeno inverso a Gemini, donde few-shot mejoró +1.5pp. El comportamiento sugiere que los 8 ejemplos cortos (80 palabras) introducen sesgo o confunden al modelo más de lo que ayudan.
- DISC en few-shot: recall altísimo (0.81) pero precision muy baja (0.28) → el modelo sobrepredicte DISC cuando ve los ejemplos de esa clase.
- LIM en few-shot: precision=1.00 pero recall=0.11 → el modelo es extremadamente conservador, probablemente porque el ejemplo de LIM en 80 palabras no captura bien la clase.
- METH se mantiene robusto en ambos modos (0.81 y 0.77). RES cae de 0.43 a 0.22 en few-shot.
- La truncación a 80 palabras fue necesaria para evitar desbordamiento de VRAM (400 palabras × 8 ejemplos ≈ 4000 tokens → offload a CPU → 283s/muestra). Con 80 palabras el modelo corre en GPU a ~11s/muestra (3× más lento que zero-shot por mayor contexto).
- CONTR = 0.00 F1 en ambos modos (solo 3 muestras en test, no conclusivo).

---

## Resumen comparativo

| Modelo | Macro F1 | Accuracy | Notas |
|---|---|---|---|
| TF-IDF + LogReg | 0.2993 | 0.4348 | Baseline, sin semántica profunda |
| SciBETO-large ft | 0.3983 | 0.4472 | Fine-tuning, early stopping en época 3 |
| Gemini 2.5 Flash (zero-shot) | 0.4730 | 0.6211 |
| Gemini 2.5 Flash (few-shot) | 0.4878 | 0.6335 |
| Qwen3-8B (zero-shot) | 0.4165 | 0.5217 | Inferencia local, greedy, sin fine-tuning |
| Qwen3-8B (few-shot) | 0.3538 | 0.4534 | Few-shot empeora (ejemplos truncados a 80 palabras) |

---

## Notas metodológicas importantes

1. **Dataset**: se usa `etiqueta_anotador` (corrección humana), NO `etiqueta_original` (heurística). El dataset de comparación interno del equipo usó `etiqueta_original` con distribución perfectamente balanceada (200/clase), por lo que los resultados no son directamente comparables.

2. **Split**: siempre document-level 80/10/10 con seed=42. Todos los modelos usan exactamente el mismo test set (161 muestras) para comparación justa.

3. **CONTR en test**: solo 3 muestras. F1 de esta clase no es estadísticamente significativo. Se documenta como limitación.

4. **Costes computacionales**:
   - TF-IDF: < 5 segundos de entrenamiento + inferencia
   - SciBETO fine-tuning: ~20 min en GPU CUDA (PyTorch 2.7.0+cu118)
   - Gemini API: ~11 min por modo (rate limit gratuito)
   - Qwen local: depende del modelo y hardware
