"""Preprocesamiento y muestreo balanceado del dataset PubMed RCT."""

import random
import re
from collections import defaultdict
from typing import Optional

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


# ---------------------------------------------------------------------------
# Limpieza editorial para fragmentos IMRaD-8 (propuesta A — Sección VIII)
# ---------------------------------------------------------------------------

# Patrones aplicados en orden. Cada uno se compila una sola vez por módulo.
_RE_URL          = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_DOI          = re.compile(r"\bdoi\s*:?\s*10\.\d{4,9}/[^\s,;]+", re.IGNORECASE)
_RE_ISSN         = re.compile(r"\bissn\s*:?\s*\d{4}-?\d{3}[\dxX]\b", re.IGNORECASE)
_RE_ISBN         = re.compile(r"\bisbn\s*:?\s*[\d\-xX]{10,17}\b", re.IGNORECASE)
# Citas estilo (Autor et al., 2019), (García y López, 2020), (García, 2018)
_RE_CITA_AUTOR   = re.compile(
    r"\((?:[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-]+"
    r"(?:\s+(?:et\s+al\.?|y|and|&)\s+[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ\-]+)?"
    r"(?:,\s*\d{4}[a-z]?)?(?:;\s*[^()]*?)?)\)"
)
# Citas numéricas estilo [12], [3, 4], [5–7]
_RE_CITA_NUM     = re.compile(r"\[\s*\d+(?:\s*[,;–\-]\s*\d+)*\s*\]")
# Referencias a tablas/figuras "vacías"
_RE_TAB_FIG      = re.compile(
    r"\b(?:tabla|cuadro|figura|fig\.?|gráfico|gráfica)\s+\d+\.?",
    re.IGNORECASE,
)
# Encabezados numerados sueltos al inicio (1., 1.2, 4.1.3, "Capítulo 2")
_RE_ENC_NUM      = re.compile(
    r"^\s*(?:cap[ií]tulo|secci[óo]n|parte)?\s*\d+(?:\.\d+){0,3}\.?\s+",
    re.IGNORECASE,
)
# Encabezados IMRaD literales al inicio (potencial leakage de la etiqueta)
_RE_ENC_IMRAD    = re.compile(
    r"^\s*(?:introducci[óo]n|antecedentes|background|metodolog[íi]a|m[ée]todos?"
    r"|materiales?\s+y\s+m[ée]todos?|resultados?|hallazgos?|discusi[óo]n"
    r"|conclusi[óo]n(?:es)?|limitaciones?|contribuciones?|aportes?)"
    r"\s*[:\.\-–]?\s*",
    re.IGNORECASE,
)
# Tokens funcionales en inglés intercalados (proxies de fragmentos bilingües)
_RE_EN_STOPWORDS = re.compile(
    r"\b(?:the|of|and|in|to|for|with|that|this|are|was|were|from|by|on|as|"
    r"which|these|those|their|there|been|have|has|had)\b",
    re.IGNORECASE,
)

_RE_PUNCT_DUP    = re.compile(r"([\.,;:])\1{1,}")
_RE_SPACE        = re.compile(r"\s+")


def clean_text_imrad8(texto: str, drop_imrad_header: bool = True,
                      drop_english_stopwords: bool = False) -> str:
    """Limpieza editorial para fragmentos IMRaD-8 (propuesta A, Sección VIII).

    Elimina ISSN/ISBN/DOI/URLs, citas (autor/numéricas), referencias a
    tablas/figuras vacías, encabezados numerados y, opcionalmente, encabezados
    IMRaD literales al inicio del fragmento (para evitar leakage de etiqueta).

    Args:
        texto: cadena original.
        drop_imrad_header: si True, retira el encabezado IMRaD al inicio.
        drop_english_stopwords: si True, elimina stopwords inglesas
            intercaladas (útil cuando el corpus tiene mezclas bilingües).

    Returns:
        Cadena limpia, con espacios colapsados y sin saltos de línea.
    """
    if not isinstance(texto, str):
        return ""
    t = texto.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    t = _RE_URL.sub(" ", t)
    t = _RE_DOI.sub(" ", t)
    t = _RE_ISSN.sub(" ", t)
    t = _RE_ISBN.sub(" ", t)
    t = _RE_CITA_AUTOR.sub(" ", t)
    t = _RE_CITA_NUM.sub(" ", t)
    t = _RE_TAB_FIG.sub(" ", t)
    if drop_imrad_header:
        t = _RE_ENC_IMRAD.sub("", t)
    t = _RE_ENC_NUM.sub("", t)
    if drop_english_stopwords:
        t = _RE_EN_STOPWORDS.sub(" ", t)
    t = _RE_PUNCT_DUP.sub(r"\1", t)
    t = _RE_SPACE.sub(" ", t).strip()
    return t


def clean_dataset_imrad8(dataset: Dataset, text_column: str = "text",
                         drop_imrad_header: bool = True,
                         drop_english_stopwords: bool = False,
                         out_column: Optional[str] = None) -> Dataset:
    """Aplica `clean_text_imrad8` a `text_column` y devuelve el dataset modificado.

    Si `out_column` es None, sobrescribe la columna original.
    """
    target = out_column or text_column

    def _map(ex):
        ex[target] = clean_text_imrad8(
            ex[text_column],
            drop_imrad_header=drop_imrad_header,
            drop_english_stopwords=drop_english_stopwords,
        )
        return ex

    return dataset.map(_map, desc=f"Limpiando IMRaD-8 → {target}")
    return dataset.map(lambda ex: {"text": clean_text(ex["text"])}, desc="Limpiando textos")
