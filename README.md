# SciDocsSpanish — Clasificación Retórica de Artículos Científicos en Español

**Proyecto final — Procesamiento de Lenguaje Natural**
Universidad de los Andes

**Integrantes:** Anderson Rodríguez · Andrés Romero · Daniel Caro · Nicolas Ríos · William Florez

**Aplicación desplegada:** [https://app.prysciencedocs.xyz](https://app.prysciencedocs.xyz)

---

## Descripción

Este proyecto aborda dos tareas de clasificación automática sobre fragmentos de artículos científicos escritos en español:

- **Tarea 1 — Clasificación retórica IMRaD (8 clases):** dado un fragmento, predice su rol estructural: `INTRO`, `BACK`, `METH`, `RES`, `DISC`, `CONC`, `CONTR` o `LIM`.
- **Tarea 2 — Detección de contribución (binario):** determina si el fragmento declara una contribución científica explícita del trabajo (`contribucion` / `no_contribucion`).

Se comparan tres paradigmas de modelado:

| Paradigma | Modelos |
|---|---|
| Baseline clásico | TF-IDF + Regresión Logística |
| Fine-tuning de encoder | SciBETO-large (fine-tuned) |
| LLM por prompting | Gemini 2.5 Flash · Qwen3-8B |

Los modelos fine-tuned están publicados en HuggingFace:
- **T1:** [wiflore/SciBETO-IMRaD](https://huggingface.co/wiflore/SciBETO-IMRaD)
- **T2:** [wiflore/SciBETO-T2-contribucion](https://huggingface.co/wiflore/SciBETO-T2-contribucion)

---

## Estructura del Repositorio

```
├── api/                  # API REST (FastAPI)
│   └── main.py
├── configs/              # Archivos de configuración YAML y prompts
│   ├── models.yaml
│   ├── labels.yaml
│   ├── data.yaml
│   └── prompts/
├── data/                 # Datos anotados y procesados
│   ├── annotated/
│   ├── mock_data/
│   └── processed/
├── models/               # Configuración local del modelo SciBETO
│   └── scibeto-large-imrad/
├── notebooks/            # Notebooks de análisis, entrenamiento y pruebas
│   ├── pipeline_imrad.ipynb
│   └── pipeline_consolidado.ipynb
├── reports/              # Resultados y reportes experimentales
│   └── entrega3/
├── scripts/              # Scripts de evaluación y utilidades
├── src/                  # Código fuente modular
│   ├── data/
│   ├── evaluation/
│   └── models/
├── .env.example          # Plantilla de variables de entorno
├── requirements.txt      # Dependencias del proyecto
└── README.md
```

---

## Dependencias y Entorno de Ejecución

**Requisitos:** Python 3.10+, pip, GPU con CUDA (recomendado para SciBETO).

```bash
# 1. Crear entorno virtual
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

---

## Configuración de Variables de Entorno

Copiar `.env.example` a `.env` y completar los valores:

```bash
cp .env.example .env
```

| Variable | Descripción | Requerida para |
|---|---|---|
| `GEMINI_API_KEY` | API key de Google AI Studio | Modelo Gemini |
| `HuggingFace` | Token de HuggingFace (lectura) | Descargar modelos privados |
| `HuggingFace_USERNAME` | Usuario de HuggingFace | Subir modelos |

Obtener clave Gemini: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## Parametrización

El sistema se configura mediante archivos YAML en `configs/`:

| Archivo | Descripción |
|---|---|
| `configs/models.yaml` | IDs de modelos, hiperparámetros de entrenamiento, batch size, etc. |
| `configs/labels.yaml` | Etiquetas de cada tarea y sus descripciones |
| `configs/data.yaml` | Rutas de datos, tamaños de split, semillas |
| `configs/prompts/` | Prompts zero-shot y few-shot para modelos LLM |

---

## Pasos de Despliegue (Local)

### Levantar la API

```bash
# Desde la raíz del repositorio
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

La API queda disponible en `http://localhost:8000`. Documentación interactiva: `http://localhost:8000/docs`.

### Endpoints principales

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/clasificar` | T1: clasifica un fragmento en 8 clases IMRaD |
| `POST` | `/contribucion` | T2: detecta si el fragmento es una contribución |
| `POST` | `/analizar` | Pipeline completo: segmenta texto y aplica T1 + T2 |
| `GET` | `/health` | Estado del servidor |
| `GET` | `/modelos` | Modelos disponibles por tarea |

**Parámetro `modelo`:** `"scibeto"` (default) · `"gemini"` · `"qwen"`

---

## Ejemplos de Uso

### Vía interfaz web

Acceder a [https://app.prysciencedocs.xyz](https://app.prysciencedocs.xyz), pegar el texto del artículo, seleccionar la familia de modelo y hacer clic en **Analizar T1 + T2**.

### Vía API (curl)

```bash
# T1: clasificar un fragmento
curl -X POST http://localhost:8000/clasificar \
  -H "Content-Type: application/json" \
  -d '{"texto": "Se diseñó un experimento controlado con 60 participantes divididos en dos grupos.", "modelo": "scibeto"}'

# Pipeline completo sobre texto largo
curl -X POST http://localhost:8000/analizar \
  -H "Content-Type: application/json" \
  -d '{"texto": "<texto completo del artículo>", "modelo": "gemini"}'
```

**Input (fragmento):**
> "Sin embargo, los resultados de este estudio están limitados por el tamaño reducido de la muestra y el uso de datos transversales que impiden establecer relaciones causales."

**Output esperado:**
```json
{
  "etiqueta": "LIM",
  "confianza": 0.97,
  "probabilidades": {"LIM": 0.97, "DISC": 0.02, ...},
  "modelo_usado": "wiflore/SciBETO-IMRaD"
}
```

---

## Reproducción de Experimentos

El notebook principal contiene todo el pipeline de entrenamiento y evaluación:

```bash
jupyter notebook notebooks/pipeline_consolidado.ipynb
```

Los resultados individuales por experimento están en `reports/entrega3/`:
- `experimentos_task1.md` — descripción y métricas de los 6 experimentos T1
- `experimentos_task2.md` — descripción y métricas de los 4 experimentos T2
- `consolidado/` — JSONs consolidados con todas las métricas

---

## Modelos y Rendimiento

| Modelo | Tarea | Macro F1 |
|---|---|---|
| TF-IDF + LR (baseline) | T1 | 0.52 |
| SciBETO fine-tuned | T1 | 0.74 |
| Gemini 2.5 Flash (v2, few-shot) | T1 | 0.65 |
| Qwen3-8B (few-shot) | T1 | 0.61 |
| SciBETO fine-tuned | T2 | 0.78 |
| Gemini 2.5 Flash | T2 | 0.71 |

---

## Trabajo Futuro

- Fine-tuning de LLMs (LoRA) sobre Qwen3-8B para clases minoritarias.
- Anotación de más ejemplos de las clases `CONTR` y `LIM`.
- Despliegue en producción con autoscaling (AWS ECS / Lambda).
