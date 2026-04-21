# Informe Entrega 1 — Modelaje — Clasificación Retórica IMRaD en Documentos Científicos en Español

**Curso:** Proyecto de Grado · Universidad de los Andes  
**Fecha:** 20 de abril de 2026  

---

## 1. Contexto y enfoque de desarrollo

El proyecto tiene como objetivo construir un sistema de clasificación retórica de fragmentos de artículos científicos en español siguiendo la estructura IMRaD (Introducción, Metodología, Resultados, Discusión). El desarrollo se estructuró en dos tareas: **Tarea 1**, clasificación de secciones IMRaD; y **Tarea 2**, identificación de contribuciones científicas (no desarrollada en esta entrega).

Dado que el pipeline de datos real —que involucra extracción de PDFs, normalización de OCR, enmascarado de encabezados y anotación manual con cálculo de acuerdo interanotador (Fleiss' κ)— tiene una complejidad y un tiempo de ejecución considerables, se adoptó una **estrategia de desarrollo paralelo**: el modelaje de la Tarea 1 avanzó sobre un conjunto de datos _mock_ mientras el pipeline de datos se consolida. Esta práctica es común en proyectos de ML para evitar bloqueos por dependencias, y la integración de ambas partes está prevista para la entrega siguiente.

El conjunto de datos utilizado es **PubMed RCT 200k** (abstractos en inglés de ensayos clínicos con etiquetas de sección), remapeado a cuatro categorías IMRaD y traducido automáticamente al español mediante un modelo de traducción neuronal. Se utilizaron 800 ejemplos balanceados (200 por clase) con splits estratificados de 640 / 80 / 80 para entrenamiento, validación y prueba respectivamente.

---

## 2. Modelos evaluados y resultados (Tarea 1)

Se compararon dos familias de enfoque: **fine-tuning supervisado** sobre SciBETO-large, y **prompting** (zero-shot y few-shot) sobre cuatro modelos de lenguaje: dos locales (Qwen2.5:3b y Qwen2.5:7b vía Ollama) y dos de API (Gemini 2.5 Flash y Gemini 3 Flash Preview de Google).

| Modelo | Estrategia | Accuracy | Macro F1 | Latencia media |
|---|---|---|---|---|
| **Gemini 2.5 Flash** | few-shot | **0.875** | **0.875** | 3.46 s/muestra |
| Gemini 3 Flash Preview | few-shot | 0.863 | 0.860 | 3.32 s/muestra |
| **SciBETO-large (fine-tuned)** | Supervisado | 0.850 | 0.845 | ~17 ms/muestra |
| Gemini 3 Flash Preview | zero-shot | 0.850 | 0.848 | 2.74 s/muestra |
| Gemini 2.5 Flash | zero-shot | 0.825 | 0.823 | 2.04 s/muestra |
| Qwen2.5:7b | zero-shot | 0.713 | 0.696 | 0.30 s/muestra |
| Qwen2.5:3b | zero-shot | 0.663 | 0.613 | 0.17 s/muestra |
| Qwen2.5:7b | few-shot | 0.663 | 0.637 | 0.31 s/muestra |
| Qwen2.5:3b | few-shot | 0.538 | 0.517 | 0.19 s/muestra |

**Gemini 2.5 Flash few-shot** lidera el ranking con 0.875 de accuracy, seguido por **Gemini 3 Flash few-shot** (0.863) y **SciBETO-large fine-tuned** (0.850). Los modelos cerrados de Google superan ligeramente al modelo fine-tuned, aunque a costa de una latencia ~200× superior y dependencia de API externa.

**SciBETO-large** sigue siendo la opción más eficiente: con sólo 17 ms/muestra alcanza un Macro F1 de 0.845, competitivo frente a las APIs comerciales. Su matriz de confusión muestra que la clase _Discusión_ es la más débil (F1=0.722, recall=0.65), con confusión hacia _Introducción_ (4 errores) y _Resultados_ (3 errores). Esto refleja que el mapeo PubMed CONCLUSIONS a Discusión introduce ruido: muchos textos etiquetados como Discusión son en realidad cierres tipo conclusión que comparten vocabulario con la apertura del artículo.

Entre los modelos abiertos (Qwen), se observan tres patrones contraintuitivos:
1. **Few-shot empeora en Qwen2.5:3b** (de 0.66 a 0.54) y también en **Qwen2.5:7b** (de 0.71 a 0.66). Los ejemplos del prompt parecen confundir más que ayudar a estos modelos, posiblemente porque el modelo se ancla en las palabras de los ejemplos y deja de razonar sobre el texto a clasificar.
2. **Discusión es sistemáticamente la peor clase** en todos los modelos de prompting (recall 15–35%), confirmando que el problema no es del clasificador sino de la ambigüedad entre CONCLUSIONS y Discusión heredada del mapeo PubMed a IMRaD.
3. **Subir de 3b a 7b en Qwen sólo mejora ~5 puntos**, mientras que el salto a Gemini Flash añade ~16 puntos adicionales: el cuello de botella es la capacidad del modelo, no el prompt.

---

## 3. Ventajas, limitaciones y trade-offs

### Fine-tuning supervisado (SciBETO)
**Ventajas:** máxima precisión, latencia mínima, comportamiento determinístico, no requiere API externa.  
**Limitaciones:** requiere datos etiquetados de calidad, el entrenamiento tarda ~30 min en MPS y el modelo no es adaptable sin reentrenamiento.  
**Trade-off:** es la opción óptima para producción si se cuenta con suficientes datos anotados; su rendimiento depende directamente de la calidad y representatividad del corpus de entrenamiento.

### Prompting (zero-shot / few-shot)
**Ventajas:** no requiere datos de entrenamiento, fácil de adaptar cambiando el prompt, útil para prototipado rápido y dominios con escasos datos etiquetados.  
**Limitaciones:** latencia alta (0.4–0.9 s/muestra), sensibilidad al diseño del prompt, variabilidad entre corridas con temperatura > 0.  
**Trade-off:** few-shot no siempre supera a zero-shot; el beneficio depende del tamaño del modelo y la calidad de los ejemplos seleccionados. Para esta tarea, el zero-shot con Qwen2.5:3b resulta sorprendentemente competitivo y sin costo adicional de diseño de ejemplos.

### Justificación del diseño de prompts

El diseño de los prompts responde a una limitación observada en las pruebas preliminares: los modelos de 3–7B clasifican mejor con señales léxicas concretas que con definiciones abstractas. Por eso, se usaron marcadores visibles para cada sección objetivo y se compararon dos variantes: zero-shot, con definiciones y reglas de desambiguación, y few-shot, igual pero con cuatro ejemplos resueltos. 

Los resultados muestran un patrón claro: en los modelos grandes, few-shot mejora el rendimiento, mientras que en los pequeños lo empeora, por lo que la elección entre ambas estrategias debe depender del modelo.  Además, las reglas para casos límite trasladan al modelo el criterio de un anotador experto y alinean la clasificación con el gold standard. 

La temperatura se fijó en 0.0 para asegurar reproducibilidad y la salida se restringió a una sola palabra para facilitar el parseo.

### Limitaciones del corpus mock
El conjunto de datos utilizado presenta tres limitaciones relevantes:
1. **Dominio y lengua:** Los textos originales son abstractos clínicos en inglés (PubMed RCT), traducidos automáticamente. El español generado puede no reflejar el estilo de redacción de artículos científicos latinoamericanos.
2. **Escala reducida:** Se usaron 200 ejemplos por clase (800 total) frente a los ≥2,000 planeados. Esto limita la generalización del fine-tuning.
3. **Sin validación humana:** El corpus no cuenta con anotación manual ni cálculo de Fleiss' κ; las etiquetas provienen del mapeo automático de PubMed a IMRaD, lo que introduce un sesgo de etiquetado no auditado.

---

## 4. Estado del pipeline de datos real

El pipeline de datos real avanzó en las siguientes etapas durante las semanas 1–2:

1. Recolección y normalización unicode de textos fuente en español
2. Mapeo de etiquetas PubMed a IMRaD y limpieza de texto
3. Construcción de splits estratificados sin solapamiento a nivel de documento
4. Diseño del esquema de anotación manual y selección de muestras

Quedan pendientes para la siguiente entrega: el enmascarado de encabezados (versión 2), la anotación manual del ≥10% de ejemplos por etiqueta, el cálculo de acuerdo interanotador, y la escala a ≥2,000 ejemplos por clase.

---

## 5. Entorno de cómputo

Todos los experimentos se ejecutaron localmente en un **MacBook Pro 16" (noviembre 2024)** con las siguientes especificaciones:

| Componente | Detalle |
|---|---|
| Chip | Apple M4 Pro |
| Memoria unificada | 48 GB |
| Acelerador | MPS (Metal Performance Shaders) |
| Sistema operativo | macOS 26.4 (25E246) |

El fine-tuning de SciBETO-large tomó aproximadamente 25 minutos en MPS para 5 épocas con 640 ejemplos. Los modelos de prompting Qwen se ejecutaron vía Ollama en CPU/MPS local, sin acceso a GPU dedicada. Las inferencias con Gemini se realizaron mediante API remota (Google AI, Tier 1 Prepay).

---

## 6. Uso de asistente de inteligencia artificial

Se utilizó **GitHub Copilot (Claude Sonnet 4.6)** como asistente de desarrollo para acelerar la escritura de scripts de ingeniería: módulos de preprocesamiento, clases de prompting y fine-tuning, y refactorización del código fuente. Las decisiones experimentales, la interpretación de resultados y la redacción del informe fueron realizadas por los autores.

Ejemplos representativos de las instrucciones utilizadas:

> *"Necesito generar un conjunto de datos mock para probar el pipeline IMRaD sin depender del corpus real. Tiene que ser balanceado (200 ejemplos por clase), estar en formato HuggingFace Dataset, con textos en español y etiquetas 0–3. ¿Cómo lo estructuro?"*

> *"El modelo Gemini devuelve respuestas truncadas con finish_reason=MAX_TOKENS cuando evalúo el conjunto de prueba completo. ¿Cómo evito el truncamiento sin cambiar la lógica del prompter?"*

---

## 7. Cambios frente al planteamiento inicial

La evaluación final no replicó exactamente la lista de modelos planteada al inicio del proyecto. En la formulación preliminar se contemplaba comparar un conjunto más amplio de modelos abiertos y cerrados; sin embargo, durante la implementación se ajustó el alcance para priorizar modelos que fueran viables de ejecutar, comparar y analizar con los recursos disponibles en esta etapa.

En particular, se mantuvo **SciBETO-large** como referencia supervisada y se evaluaron **Qwen2.5:3b** y **Qwen2.5:7b** como línea base local por su disponibilidad en Ollama y su costo de experimentación prácticamente nulo. En paralelo, se incorporaron **Gemini 2.5 Flash** y **Gemini 3 Flash Preview**, aunque este último no hacía parte del planteamiento inicial. La inclusión de Gemini 3 respondió a un cambio en la oferta de modelos disponible durante el desarrollo y a la conveniencia de contrastar un modelo más reciente bajo el mismo protocolo de evaluación.

También se decidió no hacer una exploración exhaustiva de todas las combinaciones posibles de modelos, tamaños y prompts. Esa decisión fue deliberada, no una omisión accidental. Con el corpus mock actual, los resultados ya permiten distinguir tres comportamientos claros: un modelo fine-tuned eficiente y competitivo, una familia local abierta con desempeño intermedio, y una familia de API con mejor accuracy pero mayor latencia y dependencia externa. Una vez esa señal experimental quedó establecida, seguir agregando modelos similares ofrecía un retorno decreciente frente al costo en tiempo de cómputo, uso de API y esfuerzo de análisis.

En consecuencia, el alcance de esta entrega se concentró en construir una comparación técnicamente suficiente para tomar decisiones de ingeniería, más que en agotar todo el espacio posible de modelos. La evaluación podrá ampliarse en entregas posteriores si el corpus real o nuevas restricciones del proyecto hacen necesario revisar esa decisión.

---

## 8. Trabajo pendiente

1. **Integración datos reales:** reemplazar el corpus mock por el corpus anotado, re-entrenar SciBETO y re-evaluar todos los modelos.
2. **Tarea 2:** entrenar encoder binario de contribución científica y evaluar con las mismas familias de modelo.
3. **Análisis de error cualitativo:** estudio de los errores en _Discusión_, en particular la confusión con _Introducción_ y _Resultados_ que aparece consistentemente en todos los modelos.
4. **Re-evaluación con corpus humano-validado:** comprobar si la brecha Gemini/SciBETO/Qwen se mantiene cuando las etiquetas dejan de provenir del mapeo automático de PubMed a IMRaD.
