"""Descarga y almacenamiento local del dataset PubMed 200k RCT."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


def download_pubmed_rct(
    raw_dir: str | Path,
    split: str = "train",
    force_redownload: bool = False,
) -> Dataset:
    """
    Descarga el dataset PubMed 200k RCT desde HuggingFace y lo guarda en disco.

    Args:
        raw_dir: directorio donde guardar los datos crudos.
        split: split de HuggingFace a descargar ('train', 'validation', 'test').
        force_redownload: si True, descarga aunque ya exista localmente.

    Returns:
        Dataset de HuggingFace con columnas: text, labels, uid.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    save_path = raw_dir / f"pubmed_rct_{split}"

    if save_path.exists() and not force_redownload:
        logger.info("Cargando dataset desde caché local: %s", save_path)
        return Dataset.load_from_disk(str(save_path))

    logger.info("Descargando pietrolesci/pubmed-200k-rct (split=%s)...", split)
    dataset = load_dataset("pietrolesci/pubmed-200k-rct", split=split)
    dataset.save_to_disk(str(save_path))
    logger.info("Dataset guardado en %s (%d ejemplos)", save_path, len(dataset))
    return dataset


def get_dataset_info(dataset: Dataset) -> dict:
    """Retorna estadísticas básicas del dataset."""
    label_counts = {}
    for label in dataset["labels"]:
        label_counts[label] = label_counts.get(label, 0) + 1

    return {
        "total_examples": len(dataset),
        "columns": dataset.column_names,
        "label_distribution": label_counts,
    }
