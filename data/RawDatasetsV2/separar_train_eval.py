"""
Script para separar los datasets de datasets_v2 en conjuntos de
entrenamiento y evaluación.

REGLA: No-solapamiento a nivel de documento.
- Evaluación: los registros que están en dataset_anotacion_completo.csv
- Entrenamiento: todos los registros restantes, EXCLUYENDO cualquier
  fragmento de documentos que aparecen en el conjunto de anotación.

Estructura de salida:
  datasets_entrenamiento/   ← fragmentos para entrenamiento (sin docs de anotación)
  datasets_evaluacion/      ← fragmentos seleccionados para anotación/evaluación
"""

import json
import csv
import os
from collections import defaultdict

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
DIR_DATASETS = "datasets_v2"
DIR_TRAIN = "datasets_entrenamiento"
DIR_EVAL = "datasets_evaluacion"
CSV_ANOTACION = os.path.join("anotacion_humana", "dataset_anotacion_completo.csv")
DOCS_ANOTACION_TXT = os.path.join("anotacion_humana", "documentos_anotacion.txt")

# Archivos de datasets a procesar
ARCHIVOS_DATASET = [
    "dataset_back.jsonl",
    "dataset_conc.jsonl",
    "dataset_contr.jsonl",
    "dataset_disc.jsonl",
    "dataset_intro.jsonl",
    "dataset_lim.jsonl",
    "dataset_meth.jsonl",
    "dataset_res.jsonl",
]


def cargar_chunk_ids_anotacion(ruta_csv):
    """Carga los chunk_id del CSV de anotación completo."""
    chunk_ids = set()
    with open(ruta_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunk_ids.add(row["chunk_id"])
    return chunk_ids


def cargar_docs_anotacion(ruta_txt):
    """Carga la lista de documento_id reservados para anotación."""
    docs = set()
    with open(ruta_txt, encoding="utf-8") as f:
        for linea in f:
            doc_id = linea.strip()
            if doc_id:
                docs.add(doc_id)
    return docs


def procesar_dataset(archivo, dir_datasets, dir_train, dir_eval,
                     chunk_ids_eval, docs_anotacion):
    """
    Procesa un archivo JSONL y lo separa en train/eval.

    - Evaluación: registros cuyo chunk_id está en el CSV de anotación.
    - Entrenamiento: registros cuyo documento_id NO está en docs de anotación.
    - Descartados: fragmentos de documentos de anotación que NO fueron
      seleccionados para evaluación (se descartan para evitar solapamiento).
    """
    ruta_entrada = os.path.join(dir_datasets, archivo)
    ruta_train = os.path.join(dir_train, archivo)
    ruta_eval = os.path.join(dir_eval, archivo)

    n_eval = 0
    n_train = 0
    n_descartados = 0  # Fragmentos de docs de anotación no seleccionados
    n_total = 0

    with open(ruta_entrada, encoding="utf-8") as f_in, \
         open(ruta_train, "w", encoding="utf-8") as f_train, \
         open(ruta_eval, "w", encoding="utf-8") as f_eval:

        for linea in f_in:
            n_total += 1
            registro = json.loads(linea)
            chunk_id = registro["chunk_id"]
            doc_id = registro["documento_id"]

            if chunk_id in chunk_ids_eval:
                # Este fragmento fue seleccionado para evaluación
                f_eval.write(linea)
                n_eval += 1
            elif doc_id in docs_anotacion:
                # Este fragmento pertenece a un doc de anotación pero
                # no fue seleccionado → se descarta para evitar data leakage
                n_descartados += 1
            else:
                # Documento no está en anotación → entrenamiento
                f_train.write(linea)
                n_train += 1

    return {
        "archivo": archivo,
        "total": n_total,
        "train": n_train,
        "eval": n_eval,
        "descartados": n_descartados,
    }


def main():
    print("=" * 70)
    print("SEPARACIÓN DE DATASETS: ENTRENAMIENTO vs EVALUACIÓN")
    print("=" * 70)

    # Crear directorios de salida
    os.makedirs(DIR_TRAIN, exist_ok=True)
    os.makedirs(DIR_EVAL, exist_ok=True)

    # 1. Cargar chunk_ids de anotación
    print(f"\n📋 Cargando chunk_ids desde {CSV_ANOTACION}...")
    chunk_ids_eval = cargar_chunk_ids_anotacion(CSV_ANOTACION)
    print(f"   {len(chunk_ids_eval)} chunk_ids de evaluación cargados")

    # 2. Cargar documentos de anotación
    print(f"\n📄 Cargando documentos de anotación desde {DOCS_ANOTACION_TXT}...")
    docs_anotacion = cargar_docs_anotacion(DOCS_ANOTACION_TXT)
    print(f"   {len(docs_anotacion)} documentos reservados para evaluación")

    # 3. Procesar cada dataset
    print(f"\n🔄 Procesando datasets...")
    print("-" * 70)

    resultados = []
    total_train = 0
    total_eval = 0
    total_descartados = 0
    total_registros = 0

    for archivo in ARCHIVOS_DATASET:
        resultado = procesar_dataset(
            archivo, DIR_DATASETS, DIR_TRAIN, DIR_EVAL,
            chunk_ids_eval, docs_anotacion
        )
        resultados.append(resultado)
        total_train += resultado["train"]
        total_eval += resultado["eval"]
        total_descartados += resultado["descartados"]
        total_registros += resultado["total"]

        print(f"  {archivo:25s} | Total: {resultado['total']:5d} | "
              f"Train: {resultado['train']:5d} | "
              f"Eval: {resultado['eval']:4d} | "
              f"Descartados: {resultado['descartados']:4d}")

    # 4. Resumen
    print("-" * 70)
    print(f"  {'TOTAL':25s} | Total: {total_registros:5d} | "
          f"Train: {total_train:5d} | "
          f"Eval: {total_eval:4d} | "
          f"Descartados: {total_descartados:4d}")

    # 5. Verificaciones
    print(f"\n🔍 Verificaciones:")
    print(f"   Chunk IDs esperados en eval: {len(chunk_ids_eval)}")
    print(f"   Chunk IDs encontrados en eval: {total_eval}")

    if total_eval != len(chunk_ids_eval):
        print(f"   ⚠ ADVERTENCIA: No coincide el número de registros de evaluación")
    else:
        print(f"   ✅ Todos los registros de evaluación fueron encontrados")

    # Verificar no-solapamiento a nivel de documento
    print(f"\n🔍 Verificando no-solapamiento a nivel de documento...")
    problemas = 0
    for archivo in ARCHIVOS_DATASET:
        ruta_train = os.path.join(DIR_TRAIN, archivo)
        with open(ruta_train, encoding="utf-8") as f:
            for linea in f:
                reg = json.loads(linea)
                if reg["documento_id"] in docs_anotacion:
                    print(f"   ❌ Solapamiento en {archivo}: {reg['documento_id']}")
                    problemas += 1

    if problemas == 0:
        print(f"   ✅ Sin solapamiento: ningún documento de evaluación en entrenamiento")

    # 6. Guardar resumen
    resumen = {
        "total_registros": total_registros,
        "total_train": total_train,
        "total_eval": total_eval,
        "total_descartados": total_descartados,
        "docs_anotacion": len(docs_anotacion),
        "chunk_ids_eval": len(chunk_ids_eval),
        "detalle": resultados,
    }

    ruta_resumen = os.path.join(DIR_TRAIN, "resumen_particion.json")
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"\n📊 Resumen guardado en: {ruta_resumen}")

    print(f"\n" + "=" * 70)
    print(f"✅ PROCESO COMPLETADO")
    print(f"=" * 70)
    print(f"\n📁 Entrenamiento: {os.path.abspath(DIR_TRAIN)}/")
    print(f"   {total_train} fragmentos en {len(ARCHIVOS_DATASET)} archivos")
    print(f"\n📁 Evaluación: {os.path.abspath(DIR_EVAL)}/")
    print(f"   {total_eval} fragmentos en {len(ARCHIVOS_DATASET)} archivos")
    print(f"\n🗑  Descartados (prevenir data leakage): {total_descartados} fragmentos")


if __name__ == "__main__":
    main()
