"""Visualización de resultados: matriz de confusión y comparación de modelos."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Introducción", "Metodología", "Resultados", "Discusión"]


def plot_confusion_matrix(
    cm: List[List[int]],
    label_names: List[str] = LABEL_NAMES,
    title: str = "Matriz de Confusión",
    save_path: str | Path | None = None,
):
    """Genera y opcionalmente guarda la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_arr = np.array(cm)
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Matriz de confusión guardada en %s", save_path)
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict],
    metric: str = "macro_f1",
    save_path: str | Path | None = None,
):
    """Gráfico de barras comparando modelos por métrica."""
    model_names = list(results.keys())
    values = [results[m][metric] for m in model_names]

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 5))
    bars = ax.bar(model_names, values, color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Comparación de modelos — {metric}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info("Gráfico comparativo guardado en %s", save_path)
    plt.close()
