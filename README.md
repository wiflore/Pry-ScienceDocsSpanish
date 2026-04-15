# Análisis Retórico y Extracción de Contribuciones en Artículos Científicos en Español

Proyecto de grado — Universidad de los Andes

## Descripción

Pipeline de NLP de dos etapas para el análisis automático de documentos científicos en español:

1. **Clasificación retórica** — Identifica la función discursiva de cada fragmento textual (introducción, metodología, resultados, conclusión, etc.) usando modelos encoder preentrenados en español científico (SciBETO-large) y LLMs vía prompting (GPT-5 mini, Gemini 2.5 Flash, Qwen2.5:3b, Llama3.2:3b).

2. **Detección de contribuciones científicas** — Clasifica de forma binaria qué fragmentos contienen contribuciones explícitas, enriqueciendo el contexto con las etiquetas retóricas de la primera etapa.

El corpus de entrenamiento proviene de más de 1.8 millones de documentos científicos en español extraídos de [CORE](https://core.ac.uk/).

## Objetivos

- Construir un corpus anotado con roles retóricos para artículos científicos en español.
- Entrenar y comparar clasificadores supervisados frente a LLMs en ambas tareas.
- Desplegar un demostrador interactivo (FastAPI + Streamlit) que visualice la segmentación retórica y las contribuciones detectadas.

## Stack tecnológico

`Python` · `PyTorch` · `SciBETO` · `Qwen2.5` · `Llama 3.2` · `OpenAI API` · `Gemini API` · `FastAPI` · `Streamlit` · `MLflow` · `AWS`

## Estructura del repositorio

```
├── data/
│   ├── raw/            # Corpus original de CORE
│   ├── processed/      # Texto limpio y normalizado
│   ├── annotated/      # Anotaciones manuales validadas
│   └── splits/         # Train / val / test
├── notebooks/          # EDA y experimentos exploratorios
├── src/                # Código reutilizable (preprocesamiento, modelos, métricas)
├── api/                # Backend FastAPI del demostrador
├── ui/                 # Frontend Streamlit/Gradio
├── configs/            # Hiperparámetros y templates de prompts
├── models/             # Checkpoints entrenados
├── evaluation/         # Tablas comparativas y análisis de errores
├── tests/              # Pruebas unitarias
├── mockups/            # Prototipos de interfaz
├── reports/            # Reportes de entrega
└── docs/               # Documentación y propuesta de grado
```

## Documentación

- [Propuesta de Proyecto de Grado](docs/PropuestaProyectoDeGrado.pdf)