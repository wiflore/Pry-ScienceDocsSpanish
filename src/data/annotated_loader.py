"""Carga del dataset de anotación manual revisado (esquema IMRaD-8)."""

from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import Dataset


# Mapeo código -> id IMRaD-8 (debe coincidir con configs/labels.yaml: imrad8_labels)
IMRAD8_CODE_TO_ID = {
    "INTRO": 0,
    "BACK": 1,
    "METH": 2,
    "RES": 3,
    "DISC": 4,
    "CONC": 5,
    "CONTR": 6,
    "LIM": 7,
}

IMRAD8_ID_TO_NAME = {
    0: "Introducción",
    1: "Antecedentes",
    2: "Metodología",
    3: "Resultados",
    4: "Discusión",
    5: "Conclusiones",
    6: "Contribuciones",
    7: "Limitaciones",
}

IMRAD8_LABEL_NAMES = IMRAD8_ID_TO_NAME


def load_annotated_xlsx(
    xlsx_path,
    text_column: str = "texto",
    label_column: str = "etiqueta_anotador",
    save_jsonl: Optional[Path] = None,
) -> Dataset:
    """Carga el Excel de anotación manual y devuelve un `datasets.Dataset`.

    Columnas resultantes:
      - text: texto del fragmento
      - label_code: código original (INTRO, BACK, ...)
      - imrad8_label: id entero 0–7
      - num_palabras, encabezado_seccion, documento_id, chunk_id, id (si existen)
    """
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path)

    if text_column not in df.columns:
        raise ValueError(f"Columna de texto '{text_column}' no encontrada. "
                         f"Disponibles: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"Columna de etiqueta '{label_column}' no encontrada. "
                         f"Disponibles: {list(df.columns)}")

    df = df.dropna(subset=[text_column, label_column]).copy()
    df[label_column] = df[label_column].astype(str).str.strip().str.upper()

    desconocidas = sorted(set(df[label_column]) - set(IMRAD8_CODE_TO_ID))
    if desconocidas:
        raise ValueError(f"Etiquetas desconocidas en {label_column}: {desconocidas}")

    df["text"] = df[text_column].astype(str).str.strip()
    df["label_code"] = df[label_column]
    df["imrad8_label"] = df[label_column].map(IMRAD8_CODE_TO_ID).astype(int)

    columnas = ["text", "label_code", "imrad8_label"]
    for extra in ("id", "chunk_id", "documento_id", "num_palabras", "encabezado_seccion"):
        if extra in df.columns:
            columnas.append(extra)

    df = df[columnas].reset_index(drop=True)

    if save_jsonl is not None:
        save_jsonl = Path(save_jsonl)
        save_jsonl.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(save_jsonl, orient="records", lines=True, force_ascii=False)

    return Dataset.from_pandas(df, preserve_index=False)


def clean_text(texto: str) -> str:
    """Limpia saltos de línea y colapsa espacios múltiples."""
    texto = texto.replace("\n", " ").replace("\r", " ")
    while "  " in texto:
        texto = texto.replace("  ", " ")
    return texto.strip()


def preprocess(dataset: Dataset, text_column: str = "text") -> Dataset:
    """Aplica `clean_text` a la columna de texto indicada."""
    return dataset.map(
        lambda ex: {text_column: clean_text(ex[text_column])},
        desc="Limpiando textos",
    )
