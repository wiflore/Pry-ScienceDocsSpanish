"""Cálculo y comparación de métricas para clasificación IMRaD."""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ETIQUETAS = ["Introducción", "Metodología", "Resultados", "Discusión"]


def compute_metrics(y_true: list, y_pred: list, label_names: list = ETIQUETAS) -> dict:
    """Calcula accuracy, F1 macro/micro, reporte por clase y matriz de confusión."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    por_clase = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "per_class_f1": {label_names[i]: round(por_clase[i], 4) for i in range(len(por_clase))},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0
        ),
    }


def compare_models(resultados: dict) -> str:
    """Genera una tabla de texto comparando accuracy y macro F1 de todos los modelos."""
    encabezado = f"{'Modelo':<30} {'Accuracy':>10} {'Macro F1':>10}"
    separador = "-" * len(encabezado)
    filas = [encabezado, separador]
    for nombre, metricas in sorted(resultados.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        filas.append(f"{nombre:<30} {metricas['accuracy']:>10.4f} {metricas['macro_f1']:>10.4f}")
    return "\n".join(filas)
