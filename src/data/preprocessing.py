"""Preprocesamiento y muestreo balanceado del dataset PubMed RCT."""

import random
from collections import defaultdict

from datasets import Dataset

# Mapeo PubMed → IMRaD:
# 0=BACKGROUND→0=Introducción, 1=CONCLUSIONS→3=Discusión,
# 2=METHODS→1=Metodología, 3=OBJECTIVE→0=Introducción, 4=RESULTS→2=Resultados
PUBMED_TO_IMRAD = {0: 0, 1: 3, 2: 1, 3: 0, 4: 2}

NOMBRES_IMRAD = {
    0: "Introducción",
    1: "Metodología",
    2: "Resultados",
    3: "Discusión",
}

# Alias retrocompatible
IMRAD_LABEL_NAMES = NOMBRES_IMRAD


def map_pubmed_labels_to_imrad(dataset: Dataset) -> Dataset:
    """Agrega columna 'imrad_label' convirtiendo etiquetas PubMed (0-4) a IMRaD (0-3)."""
    def _map(ejemplo):
        ejemplo["imrad_label"] = PUBMED_TO_IMRAD[ejemplo["labels"]]
        return ejemplo
    return dataset.map(_map, desc="Mapeando etiquetas PubMed→IMRaD")


def sample_balanced(dataset: Dataset, examples_per_label: int, seed: int = 42) -> Dataset:
    """Devuelve un subconjunto balanceado con examples_per_label muestras por categoría IMRaD."""
    random.seed(seed)

    indices_por_etiqueta = defaultdict(list)
    for i, etiqueta in enumerate(dataset["imrad_label"]):
        indices_por_etiqueta[etiqueta].append(i)

    seleccionados = []
    for eid in range(4):
        disponibles = indices_por_etiqueta[eid]
        n = min(len(disponibles), examples_per_label)
        seleccionados.extend(random.sample(disponibles, n))

    random.shuffle(seleccionados)
    return dataset.select(seleccionados)


def clean_text(texto: str) -> str:
    """Elimina saltos de línea y colapsa espacios múltiples."""
    texto = texto.replace("\n", " ").replace("\r", " ")
    while "  " in texto:
        texto = texto.replace("  ", " ")
    return texto.strip()


def preprocess_dataset(dataset: Dataset) -> Dataset:
    """Aplica limpieza de texto al campo 'text'."""
    return dataset.map(lambda ex: {"text": clean_text(ex["text"])}, desc="Limpiando textos")
