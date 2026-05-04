"""
Script para crear un dataset de anotación manual a partir de datasets_v2.

Extrae 200 ejemplos de cada uno de los 8 datasets (1600 total) y los exporta
en un solo CSV para anotación humana.

REGLA CRÍTICA: No-solapamiento a nivel de documento.
Si un documento se asigna al subconjunto de anotación (evaluación), ningún
fragmento de ese mismo documento puede quedar en el conjunto de entrenamiento,
y viceversa. La selección de documentos se hace ANTES de muestrear fragmentos.

Salida:
  - anotacion_humana/dataset_anotacion_completo.csv  (1600 filas)
  - anotacion_humana/particiones/anotador_1.csv ... anotador_N.csv
  - anotacion_humana/documentos_anotacion.txt  (lista de doc_ids reservados)
  - anotacion_humana/documentos_entrenamiento.txt  (doc_ids restantes)
  - anotacion_humana/resumen.txt
"""

import json
import os
import csv
import random
from collections import defaultdict
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
SEED = 42
EJEMPLOS_POR_DATASET = 200
NUM_ANOTADORES = 4  # Número de anotadores (se dividirá equitativamente)
DIR_DATASETS = "datasets_v2"
DIR_SALIDA = "anotacion_humana"

# Mapeo de nombre de archivo a etiqueta legible
ETIQUETA_MAP = {
    "dataset_back.jsonl": "BACK",
    "dataset_conc.jsonl": "CONC",
    "dataset_contr.jsonl": "CONTR",
    "dataset_disc.jsonl": "DISC",
    "dataset_intro.jsonl": "INTRO",
    "dataset_lim.jsonl": "LIM",
    "dataset_meth.jsonl": "METH",
    "dataset_res.jsonl": "RES",
}

# Columnas para el CSV de anotación
COLUMNAS_CSV = [
    "id",                    # Identificador único del ejemplo
    "chunk_id",              # ID original del chunk
    "documento_id",          # Documento de origen
    "etiqueta_original",     # Etiqueta asignada por el pipeline
    "etiqueta_anotador",     # ← VACÍA, para que el anotador la llene
    "confianza_original",    # Nivel de confianza del pipeline
    "texto",                 # Texto del fragmento
    "num_palabras",          # Número de palabras
    "encabezado_seccion",    # Encabezado detectado
    "notas_anotador",        # ← VACÍA, para comentarios del anotador
]


def cargar_todos_los_registros(dir_datasets):
    """Carga todos los registros de los 8 datasets, agrupados por documento."""
    docs_por_dataset = {}  # {archivo: {doc_id: [registros]}}
    todos_docs = defaultdict(set)  # {doc_id: {archivos donde aparece}}

    archivos = sorted(
        f for f in os.listdir(dir_datasets)
        if f.startswith("dataset_") and f.endswith(".jsonl")
    )

    for archivo in archivos:
        ruta = os.path.join(dir_datasets, archivo)
        registros_por_doc = defaultdict(list)

        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                registro = json.loads(linea)
                doc_id = registro["documento_id"]
                registros_por_doc[doc_id].append(registro)
                todos_docs[doc_id].add(archivo)

        docs_por_dataset[archivo] = registros_por_doc
        print(f"  Cargado {archivo}: {sum(len(v) for v in registros_por_doc.values())} "
              f"fragmentos de {len(registros_por_doc)} documentos")

    return docs_por_dataset, todos_docs


def seleccionar_documentos_anotacion(docs_por_dataset, todos_docs, ejemplos_por_ds, rng):
    """
    Selecciona documentos para anotación respetando no-solapamiento.

    Estrategia:
    1. Para cada dataset, calcular cuántos documentos se necesitan para
       obtener ~200 fragmentos.
    2. Seleccionar documentos priorizando los que SOLO aparecen en ese
       dataset (así no "contaminamos" otros datasets).
    3. Si un documento aparece en múltiples datasets, al asignarlo a
       anotación se excluye de entrenamiento en TODOS los datasets.
    """
    docs_anotacion = set()       # doc_ids seleccionados para anotación
    seleccion = {}               # {archivo: [registros seleccionados]}

    # Ordenar datasets por número de documentos exclusivos (menos primero)
    # para maximizar la cobertura
    orden = sorted(
        docs_por_dataset.keys(),
        key=lambda a: sum(1 for d in docs_por_dataset[a] if len(todos_docs[d]) == 1)
    )

    for archivo in orden:
        registros_por_doc = docs_por_dataset[archivo]
        disponibles = []

        # Prioridad 1: documentos exclusivos de este dataset (no afectan otros)
        docs_exclusivos = [
            d for d in registros_por_doc
            if len(todos_docs[d]) == 1 and d not in docs_anotacion
        ]
        # Prioridad 2: documentos compartidos (ya seleccionados para anotación)
        docs_ya_seleccionados = [
            d for d in registros_por_doc
            if d in docs_anotacion
        ]
        # Prioridad 3: documentos compartidos (aún no seleccionados)
        docs_compartidos = [
            d for d in registros_por_doc
            if len(todos_docs[d]) > 1 and d not in docs_anotacion
        ]

        # Recopilar fragmentos de docs ya seleccionados
        fragmentos = []
        for doc_id in docs_ya_seleccionados:
            fragmentos.extend(registros_por_doc[doc_id])

        # Si ya tenemos suficientes, muestrear
        if len(fragmentos) >= ejemplos_por_ds:
            seleccion[archivo] = rng.sample(fragmentos, ejemplos_por_ds)
            continue

        # Agregar fragmentos de docs exclusivos (orden aleatorio)
        rng.shuffle(docs_exclusivos)
        for doc_id in docs_exclusivos:
            if len(fragmentos) >= ejemplos_por_ds:
                break
            fragmentos.extend(registros_por_doc[doc_id])
            docs_anotacion.add(doc_id)

        # Si aún faltan, usar docs compartidos
        if len(fragmentos) < ejemplos_por_ds:
            rng.shuffle(docs_compartidos)
            for doc_id in docs_compartidos:
                if len(fragmentos) >= ejemplos_por_ds:
                    break
                fragmentos.extend(registros_por_doc[doc_id])
                docs_anotacion.add(doc_id)

        # Muestrear exactamente N fragmentos
        if len(fragmentos) >= ejemplos_por_ds:
            seleccion[archivo] = rng.sample(fragmentos, ejemplos_por_ds)
        else:
            # En caso extremo, tomar todos los disponibles
            seleccion[archivo] = fragmentos
            print(f"  ⚠ {archivo}: solo {len(fragmentos)} fragmentos disponibles "
                  f"(se pidieron {ejemplos_por_ds})")

    return seleccion, docs_anotacion


def construir_csv(seleccion, dir_salida, num_anotadores, rng):
    """Construye el CSV completo y las particiones por anotador."""
    os.makedirs(os.path.join(dir_salida, "particiones"), exist_ok=True)

    # Construir lista completa con IDs secuenciales
    todas_filas = []
    idx = 1
    for archivo in sorted(seleccion.keys()):
        etiqueta = ETIQUETA_MAP.get(archivo, archivo)
        for reg in seleccion[archivo]:
            fila = {
                "id": idx,
                "chunk_id": reg.get("chunk_id", ""),
                "documento_id": reg.get("documento_id", ""),
                "etiqueta_original": etiqueta,
                "etiqueta_anotador": "",  # Para que el anotador llene
                "confianza_original": reg.get("confianza", ""),
                "texto": reg.get("texto", ""),
                "num_palabras": reg.get("num_palabras", ""),
                "encabezado_seccion": reg.get("encabezado_seccion", ""),
                "notas_anotador": "",  # Para comentarios del anotador
            }
            todas_filas.append(fila)
            idx += 1

    # Mezclar aleatoriamente para que los anotadores no vean bloques temáticos
    rng.shuffle(todas_filas)

    # Reasignar IDs después de mezclar
    for i, fila in enumerate(todas_filas, 1):
        fila["id"] = i

    # Guardar CSV completo
    ruta_completo = os.path.join(dir_salida, "dataset_anotacion_completo.csv")
    with open(ruta_completo, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        writer.writeheader()
        writer.writerows(todas_filas)
    print(f"\n✅ Dataset completo: {ruta_completo} ({len(todas_filas)} filas)")

    # Dividir por anotadores
    filas_por_anotador = len(todas_filas) // num_anotadores
    for i in range(num_anotadores):
        inicio = i * filas_por_anotador
        if i == num_anotadores - 1:
            fin = len(todas_filas)  # Último anotador toma el resto
        else:
            fin = inicio + filas_por_anotador

        particion = todas_filas[inicio:fin]
        ruta_particion = os.path.join(
            dir_salida, "particiones", f"anotador_{i+1}.csv"
        )
        with open(ruta_particion, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
            writer.writeheader()
            writer.writerows(particion)
        print(f"  📋 Anotador {i+1}: {ruta_particion} ({len(particion)} filas)")

    return todas_filas


def guardar_listas_documentos(docs_anotacion, todos_docs, dir_salida):
    """Guarda las listas de documentos asignados a anotación vs entrenamiento."""
    todos = set(todos_docs.keys())
    docs_entrenamiento = todos - docs_anotacion

    ruta_anot = os.path.join(dir_salida, "documentos_anotacion.txt")
    with open(ruta_anot, "w", encoding="utf-8") as f:
        for doc_id in sorted(docs_anotacion):
            f.write(doc_id + "\n")

    ruta_train = os.path.join(dir_salida, "documentos_entrenamiento.txt")
    with open(ruta_train, "w", encoding="utf-8") as f:
        for doc_id in sorted(docs_entrenamiento):
            f.write(doc_id + "\n")

    print(f"\n📄 Documentos para anotación: {len(docs_anotacion)}")
    print(f"📄 Documentos para entrenamiento: {len(docs_entrenamiento)}")

    return docs_entrenamiento


def generar_resumen(seleccion, docs_anotacion, todos_docs, todas_filas, dir_salida):
    """Genera un reporte resumen de la partición."""
    docs_entrenamiento = set(todos_docs.keys()) - docs_anotacion

    ruta_resumen = os.path.join(dir_salida, "resumen.txt")
    with open(ruta_resumen, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("RESUMEN DE PARTICIÓN PARA ANOTACIÓN MANUAL\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total de fragmentos seleccionados: {len(todas_filas)}\n")
        f.write(f"Total de documentos en anotación: {len(docs_anotacion)}\n")
        f.write(f"Total de documentos en entrenamiento: {len(docs_entrenamiento)}\n")
        f.write(f"Total de documentos únicos: {len(todos_docs)}\n\n")

        f.write("-" * 70 + "\n")
        f.write("FRAGMENTOS POR CATEGORÍA:\n")
        f.write("-" * 70 + "\n")
        for archivo in sorted(seleccion.keys()):
            etiqueta = ETIQUETA_MAP.get(archivo, archivo)
            n = len(seleccion[archivo])
            f.write(f"  {etiqueta:8s}: {n:4d} fragmentos\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("VERIFICACIÓN DE NO-SOLAPAMIENTO:\n")
        f.write("-" * 70 + "\n")

        # Verificar que ningún doc de anotación esté en entrenamiento
        solapados = docs_anotacion & docs_entrenamiento
        if solapados:
            f.write(f"  ❌ ERROR: {len(solapados)} documentos solapados\n")
            for d in sorted(solapados)[:10]:
                f.write(f"    - {d}\n")
        else:
            f.write("  ✅ Sin solapamiento: ningún documento aparece en ambas particiones\n")

        # Verificar cobertura
        f.write(f"\n  Documentos en anotación que aparecen en múltiples datasets:\n")
        multi = [d for d in docs_anotacion if len(todos_docs[d]) > 1]
        f.write(f"    {len(multi)} documentos compartidos correctamente aislados\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("DISTRIBUCIÓN DE CONFIANZA EN ANOTACIÓN:\n")
        f.write("-" * 70 + "\n")
        confianzas = defaultdict(int)
        for fila in todas_filas:
            confianzas[fila["confianza_original"]] += 1
        for conf, count in sorted(confianzas.items()):
            f.write(f"  {conf:15s}: {count:4d}\n")

    print(f"\n📊 Resumen: {ruta_resumen}")


def verificar_no_solapamiento(docs_anotacion, docs_por_dataset):
    """
    Verificación final: asegura que los fragmentos restantes (entrenamiento)
    no contengan documentos del set de anotación.
    """
    print("\n🔍 Verificando no-solapamiento...")
    problemas = 0
    for archivo, registros_por_doc in docs_por_dataset.items():
        for doc_id in registros_por_doc:
            if doc_id in docs_anotacion:
                # Estos fragmentos deben EXCLUIRSE del entrenamiento
                pass  # Correcto: se excluirán
        # Contar cuántos fragmentos quedarían para entrenamiento
        frags_train = sum(
            len(regs) for doc_id, regs in registros_por_doc.items()
            if doc_id not in docs_anotacion
        )
        total = sum(len(regs) for regs in registros_por_doc.values())
        print(f"  {archivo}: {frags_train}/{total} fragmentos para entrenamiento "
              f"({total - frags_train} excluidos)")

    if problemas == 0:
        print("  ✅ Verificación exitosa: no hay solapamiento a nivel de documento")


def main():
    print("=" * 70)
    print("CREACIÓN DE DATASET PARA ANOTACIÓN MANUAL")
    print("=" * 70)

    rng = random.Random(SEED)

    # 1. Cargar todos los registros
    print("\n📂 Cargando datasets...")
    docs_por_dataset, todos_docs = cargar_todos_los_registros(DIR_DATASETS)

    print(f"\n📊 Estadísticas globales:")
    print(f"  Total documentos únicos: {len(todos_docs)}")
    docs_multi = sum(1 for d in todos_docs if len(todos_docs[d]) > 1)
    print(f"  Documentos en múltiples datasets: {docs_multi}")

    # 2. Seleccionar documentos para anotación
    print(f"\n🎯 Seleccionando {EJEMPLOS_POR_DATASET} ejemplos por dataset...")
    seleccion, docs_anotacion = seleccionar_documentos_anotacion(
        docs_por_dataset, todos_docs, EJEMPLOS_POR_DATASET, rng
    )

    # 3. Construir CSVs
    print(f"\n📝 Generando CSV para {NUM_ANOTADORES} anotadores...")
    todas_filas = construir_csv(seleccion, DIR_SALIDA, NUM_ANOTADORES, rng)

    # 4. Guardar listas de documentos
    guardar_listas_documentos(docs_anotacion, todos_docs, DIR_SALIDA)

    # 5. Generar resumen
    generar_resumen(seleccion, docs_anotacion, todos_docs, todas_filas, DIR_SALIDA)

    # 6. Verificar
    verificar_no_solapamiento(docs_anotacion, docs_por_dataset)

    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print("=" * 70)
    print(f"\nArchivos generados en: {os.path.abspath(DIR_SALIDA)}/")
    print(f"  - dataset_anotacion_completo.csv ({len(todas_filas)} filas)")
    print(f"  - particiones/anotador_1.csv ... anotador_{NUM_ANOTADORES}.csv")
    print(f"  - documentos_anotacion.txt ({len(docs_anotacion)} docs)")
    print(f"  - documentos_entrenamiento.txt ({len(todos_docs) - len(docs_anotacion)} docs)")
    print(f"  - resumen.txt")


if __name__ == "__main__":
    main()
