# Clasificación Retórica IMRaD de Fragmentos Científicos en Español

Proyecto final — Maestría en Inteligencia Artificial, Universidad de los Andes

**Integrantes:** William Florez, [Co-Autor 2]

## Descripción

Este proyecto aborda la clasificación retórica automática de fragmentos de artículos científicos en español utilizando un esquema extendido IMRaD de 8 clases (INTRO, BACK, METH, RES, DISC, CONC, CONTR, LIM). Se comparan tres paradigmas de modelado:
1. Baseline clásico (TF-IDF + LR)
2. Fine-tuning de encoder de dominio (SciBETO-large)
3. LLMs por prompting (Gemini 3 Flash, Qwen3-8B).

El reporte final completo está disponible en [`reports/entrega_final/reporte_final.md`](reports/entrega_final/reporte_final.md).

## Requisitos y Entorno

Para reproducir los experimentos, se requiere un entorno Python 3.9+ (preferiblemente Apple Silicon M-series para Qwen/MLX).

```bash
# Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Estructura de Archivos Principal

*   `notebooks/pipeline_consolidado.ipynb`: **Notebook principal**. Contiene la carga de datos, splits, evaluación del baseline TF-IDF, fine-tuning de SciBETO y evaluación zero-shot de Gemini/Qwen.
*   `reports/entrega_final/reporte_final.md`: Artículo académico formato IEEE con análisis detallado.

## Instrucciones de Uso y Reproducción

### 1. Ejecutar el Pipeline Consolidado
Toda la lógica base se encuentra en el Jupyter Notebook:
```bash
jupyter notebook notebooks/pipeline_consolidado.ipynb
```
El notebook está diseñado para correr de principio a fin de manera secuencial, generando las métricas reportadas en el documento final.

## Ejemplos de Input / Output

**Input (Fragmento de artículo científico):**
> "Sin embargo, los resultados de este estudio están limitados por el tamaño reducido de la muestra y el uso de datos transversales que impiden establecer relaciones causales."

**Output esperado:**
> `LIM` (Limitaciones)

## Consideraciones de Modelos y Despliegue

*   **SciBETO-large:** Requiere ~1.5 GB de memoria VRAM. Es ideal para procesamiento masivo offline y on-premises. Latencia: <10ms por fragmento en GPU.
*   **Gemini 3 Flash:** No requiere infraestructura propia, operado vía API. Latencia: 1–3s por petición HTTP. Excelente costo/beneficio en bajo volumen.
*   **Qwen3-8B (4-bit):** Requiere ~5.5 GB de memoria unificada en dispositivos Apple. Ejecutado vía MLX, demostrando que es posible operar modelos Llama3/Qwen3 en entornos limitados y privados.

## Enlaces
*   **Checkpoint SciBETO-IMRaD:** [wiflore/SciBETO-IMRaD (HuggingFace)](https://huggingface.co/wiflore/SciBETO-IMRaD)

## Trabajo Futuro
*   **Fine-tuning de LLMs (LoRA):** Como trabajo futuro por fuera del alcance actual, se propone explorar el ajuste fino de modelos fundacionales abiertos (como Qwen3-8B) mediante Low-Rank Adaptation (LoRA) para buscar mejorar las métricas en clases minoritarias manteniendo control sobre los pesos del modelo.
