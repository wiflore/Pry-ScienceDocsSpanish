# Experimentos — Tarea 2: Extracción de Contribuciones Científicas (binario)

**Proyecto:** Análisis de documentos científicos en español  
**Grupo:** FLAG – TICsW, Universidad de los Andes  
**Última actualización:** 2026-05-13 (Experimento 4 — SciBETO fine-tuned)

---

## Configuración experimental común

### Dataset
| Parámetro | Valor |
|---|---|
| Fuente | `data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl` |
| Total candidatos | 2 000 |
| Etiqueta usada | `label` (heurística curada: 1=contribucion, 0=no_contribucion) |
| Split | Predefinido por campo `split` |
| Train (`train_weak`) | 1 600 (800 pos / 800 neg) |
| Val (`val_weak`) | 200 (100 pos / 100 neg) |
| Test (`gold_eval_pendiente`) | 200 (100 pos / 100 neg, balanceado) |

> **Nota:** El test gold_eval está **balanceado** (50/50), lo que hace las métricas más interpretables que la distribución real del corpus (~3% positivos). Los resultados son comparables entre sí pero no directamente con el protocolo del compañero, que evaluó sobre la distribución natural del split Tarea 1 (165 muestras, sólo 5 positivos).

### Definición de la tarea
Un fragmento se clasifica como `contribucion` (label=1) si declara **explícitamente** un aporte científico propio del trabajo: método propuesto, sistema desarrollado, hallazgo original, corpus construido, etc. La contribución debe estar atribuida al propio trabajo/artículo.

### Métrica principal
**F1 de la clase positiva** (`contribucion`, label=1). Se reportan también Precisión, Recall y F1-macro. El F1-macro sobre un test balanceado es equivalente a la media aritmética de F1-pos y F1-neg.

---

## Experimento 1 — Baseline: TF-IDF + Regresión Logística

**Script:** `scripts/eval_task2_tfidf.py`  
**Fecha:** 2026-05-13  
**Estado:** ✅ Completado

### Configuración
| Parámetro | Valor |
|---|---|
| Vectorizador | TF-IDF, ngram=(1,2), max_features=100 000, sublinear_tf=True, min_df=2 |
| Clasificador | Logistic Regression, C=1.0, max_iter=1000, solver=lbfgs |
| Pesos de clase | Inverso de frecuencia (class_weight=dict) |

### Resultados en test
| Métrica | Valor |
|---|---|
| **F1-pos (contribucion)** | **0.7739** |
| Precisión (contribucion) | 0.6846 |
| Recall (contribucion) | 0.8900 |
| **F1-macro** | **0.7340** |
| Predicciones positivas | 130 / 100 (sobre-predice) |

#### Por clase
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| no_contribucion | 0.84 | 0.59 | 0.69 | 100 |
| **contribucion** | **0.68** | **0.89** | **0.77** | 100 |

#### Matriz de confusión
```
                   pred_neg  pred_pos
real_neg (100)        59        41
real_pos (100)        11        89
```

### Análisis
El TF-IDF con pesos de clase tiene recall alto (0.89) pero precisión baja (0.68), produciendo 41 falsos positivos. El modelo aprende las palabras más discriminativas de los fragmentos de contribución ("proponemos", "presentamos", "desarrollamos") pero se confunde con fragmentos que usan vocabulario similar en contexto no contributivo.

---

## Experimento 2 — Gemini 2.5 Flash (zero-shot y few-shot)

**Script:** `scripts/eval_task2_gemini.py`  
**Fecha:** 2026-05-13  
**Estado:** ✅ Completado

### Configuración
| Parámetro | Valor |
|---|---|
| Modelo | `gemini-2.5-flash` |
| Temperatura | 0.0 |
| max_output_tokens | 15 |
| thinking_budget | 0 (sin CoT) |
| Few-shot | 1 ejemplo por clase, truncado a 400 palabras |
| Rate limit delay | 4.2 s entre requests |
| Prompt | Definición binaria detallada + criterios de exclusión |

### Resultados — Zero-shot
| Métrica | Valor |
|---|---|
| **F1-pos** | **0.7101** |
| Precisión | 0.8696 |
| Recall | 0.6000 |
| **F1-macro** | **0.7490** |
| Predicciones positivas | 69 / 100 (sub-predice) |
| Inválidos | 0 |

### Resultados — Few-shot
| Métrica | Valor |
|---|---|
| **F1-pos** | **0.8000** |
| Precisión | 0.8444 |
| Recall | 0.7600 |
| **F1-macro** | **0.8095** |
| Predicciones positivas | 90 / 100 |
| Inválidos | 0 |

### Análisis
A diferencia de Tarea 1 (donde few-shot fue marginal), en Tarea 2 el few-shot **mejora F1-pos en +9 pp** (0.71 → 0.80). El zero-shot es demasiado conservador (recall=0.60, solo 69 predicciones positivas de 100). Los ejemplos concretos ayudan a Gemini a calibrar su umbral interno para "contribución". La precisión sigue siendo alta en ambos modos (0.87 / 0.84), lo que indica que cuando el modelo predice positivo, acierta con alta frecuencia.

---

## Experimento 3 — Qwen3-8B (zero-shot y few-shot)

**Script:** `scripts/eval_task2_qwen.py`  
**Fecha:** 2026-05-13  
**Estado:** ✅ Completado  
**Hardware:** NVIDIA RTX 4070, bfloat16, HuggingFace Transformers

### Configuración
| Parámetro | Valor |
|---|---|
| Modelo | `Qwen/Qwen3-8B` |
| Thinking | Desactivado (`/no_think` + `enable_thinking=False`) |
| max_new_tokens | 15 |
| Decoding | Greedy (do_sample=False) |
| Few-shot | 1 ejemplo por clase, truncado a **80 palabras** (límite VRAM) |
| Prompt | Definición binaria detallada + criterios de exclusión |

### Resultados — Zero-shot
| Métrica | Valor |
|---|---|
| **F1-pos** | **0.6979** |
| Precisión | 0.7283 |
| Recall | 0.6700 |
| **F1-macro** | **0.7095** |
| Predicciones positivas | 92 / 100 |
| Inválidos | 0 |

### Resultados — Few-shot
| Métrica | Valor |
|---|---|
| **F1-pos** | **0.7489** |
| Precisión | 0.6693 |
| Recall | 0.8500 |
| **F1-macro** | **0.7097** |
| Predicciones positivas | 127 / 100 (sobre-predice) |
| Inválidos | 0 |

### Análisis
Contrariamente a Tarea 1 (donde few-shot empeoró a Qwen), en Tarea 2 el few-shot **mejora F1-pos en +5 pp** (0.698 → 0.749). Sin embargo, los ejemplos de 80 palabras hacen que el modelo sea más permisivo: recall sube de 0.67 a 0.85 pero precisión baja de 0.73 a 0.67. El patrón es consistente con que los 80 palabras de ejemplo son suficientes para señalar "qué tipo de texto es una contribución" pero no para enseñar el límite de lo que no lo es. Qwen queda por debajo de Gemini en ambos modos.

---

## Experimento 4 — SciBETO-large (fine-tuning binario)

**Script:** `scripts/train_task2_scibeto.py`  
**Fecha:** 2026-05-13  
**Estado:** ✅ Completado  
**Hardware:** NVIDIA RTX 4070, float32, HuggingFace Transformers  
**Modelo guardado:** `models/scibeto-task2-binario/best_model/`

### Configuración
| Parámetro | Valor |
|---|---|
| Modelo base | `Flaglab/SciBETO-large` |
| Tokenización | Head+Tail (128 + 382 = 510 + CLS/SEP = 512) |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Scheduler | Linear warmup (10%) + decay |
| Weight decay | 0.01 |
| Classifier dropout | 0.3 |
| Max epochs | 10 |
| Early stopping | patience=3, métrica=F1-pos en val |
| Pesos de clase | 1.0 / 1.0 (dataset balanceado → no necesarios) |

### Entrenamiento
| Época | Train Loss | Val Loss | Val F1-pos |
|---|---|---|---|
| 1 | 0.5527 | 0.3324 | 0.8899 |
| 2 | 0.3223 | 0.4171 | 0.8739 |
| **3** | **0.2391** | **0.4845** | **0.9108** ← mejor |
| 4 | 0.1561 | 0.5554 | 0.9073 |
| 5 | 0.0862 | 0.6719 | 0.9100 |
| 6 | 0.0242 | 0.6408 | 0.9064 |
| — | — | — | Early stop (patience=3) |

### Resultados en test
| Métrica | Valor |
|---|---|
| **F1-pos (contribucion)** | **0.8426** |
| Precisión (contribucion) | 0.7333 |
| Recall (contribucion) | **0.9900** |
| **F1-macro** | **0.8092** |
| Predicciones positivas | 135 / 100 (sobre-predice) |

#### Por clase
| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| no_contribucion | 0.98 | 0.64 | 0.78 | 100 |
| **contribucion** | **0.73** | **0.99** | **0.84** | 100 |

#### Matriz de confusión
```
                   pred_neg  pred_pos
real_neg (100)        64        36
real_pos (100)         1        99
```

### Análisis
SciBETO es el mejor modelo de Tarea 2 (F1-pos=**0.8426**). El recall de **0.99** es extraordinario: sólo 1 contribución real fue clasificada como no-contribución. El costo es una precisión más baja (0.73): 36 falsos positivos de 135 predicciones positivas. El dominio pre-entrenado en español científico da ventaja clara sobre Qwen3 y el baseline TF-IDF. El early stopping paró en época 6 con el mejor checkpoint en época 3 (val_f1=0.9108), señal de que el modelo aprende rápido y luego sobreajusta.

---

## Resumen comparativo

| Modelo | F1-pos | Precisión | Recall | F1-macro | Pred. pos. |
|---|---|---|---|---|---|
| Qwen3-8B zero-shot | 0.6979 | 0.7283 | 0.6700 | 0.7095 | 92 |
| Gemini 2.5 Flash zero-shot | 0.7101 | **0.8696** | 0.6000 | 0.7490 | 69 |
| Qwen3-8B few-shot | 0.7489 | 0.6693 | 0.8500 | 0.7097 | 127 |
| TF-IDF + LogReg | 0.7739 | 0.6846 | 0.8900 | 0.7340 | 130 |
| Gemini 2.5 Flash few-shot | 0.8000 | 0.8444 | 0.7600 | 0.8095 | 90 |
| **SciBETO fine-tuned** | **0.8426** | 0.7333 | **0.9900** | **0.8092** | 135 |

> Test set: `gold_eval_pendiente`, 200 muestras, 100 pos / 100 neg (balanceado).

---

## Análisis comparativo

### ¿Por qué SciBETO domina en Tarea 2?
Al igual que en Tarea 1, el preentrenamiento en español científico da a SciBETO una representación semántica que le permite distinguir con alta fidelidad fragmentos de contribución. Con 1 600 ejemplos de entrenamiento balanceados y fine-tuning supervisado, el modelo aprende un umbral de decisión muy ajustado: recall=0.99 con solo 1 falso negativo.

### ¿Por qué few-shot ayuda en Tarea 2 pero no en Tarea 1?
En Tarea 1 (8 clases) los ejemplos de 80 palabras contenían demasiada señal ruidosa al truncar fragmentos complejos. En Tarea 2 (binario) la señal es más simple: los ejemplos le muestran al modelo exactamente qué tipo de declaración cuenta como contribución. Con sólo 2 clases, incluso ejemplos cortos son informativos.

### ¿Por qué Gemini few-shot supera a Qwen few-shot?
Gemini tiene mayor ventana de contexto efectiva y puede procesar los 400 palabras del ejemplo sin degradación. Qwen usa solo 80 palabras, lo que hace sus ejemplos menos representativos. La diferencia de F1-pos es de +5 pp (0.80 vs 0.749).

### ¿Por qué TF-IDF supera a Qwen zero-shot?
Con un dataset balanceado y vocabulario discriminativo fuerte, TF-IDF captura las señales léxicas directas de contribución ("proponemos", "presentamos", "desarrollamos un nuevo"). Qwen zero-shot sin ejemplos es más incierto sobre dónde trazar el límite entre contribución y no-contribución.

---

## Notas metodológicas

- **Comparabilidad con protocolo del compañero:** El compañero evaluó sobre distribución natural (~3% positivos, 5 positivos de 165 muestras) y obtuvo F1-pos=0 (TF-IDF) y F1-pos=0.094 (Gemini). Nuestro protocolo usa el gold eval balanceado (50/50). Las métricas **no son directamente comparables** pero el ranking relativo de los modelos es informativo.
- **Etiqueta heurística vs manual:** El campo `etiqueta_manual` está vacío en el gold eval. Se usa `label` (generado heurísticamente y curado). La anotación manual pendiente podría alterar hasta ~10% de las etiquetas.
- **Umbral de decisión:** SciBETO con recall=0.99 puede ser preferible en aplicaciones donde no se quiere perder ninguna contribución (alta cobertura), aceptando el ruido de los 36 falsos positivos. Gemini few-shot con precisión=0.84 es preferible si se quiere alta fidelidad en las predicciones positivas.
