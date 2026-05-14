"""
Baseline TF-IDF + Logistic Regression para Tarea 2: extracción de contribuciones
científicas (clasificación binaria: contribucion / no_contribucion).

Dataset: data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl
  - split='train_weak'           -> entrenamiento (1600 muestras)
  - split='val_weak'             -> validación (200 muestras)
  - split='gold_eval_pendiente'  -> test (200 muestras, 100/100 balanceado)

Métrica principal: F1 de la clase positiva (label=1, 'contribucion').
"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 42
DATA_PATH    = Path("data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl")
RESULTS_PATH = Path("reports/entrega3/results_task2_tfidf.json")

LABEL_NAMES = ["no_contribucion", "contribucion"]  # 0, 1


def load_split(data: list[dict], split_name: str):
    rows = [d for d in data if d["split"] == split_name]
    X = [d["texto"] for d in rows]
    y = [d["label"] for d in rows]
    return X, y


def main():
    np.random.seed(SEED)

    # 1. Cargar datos
    data = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]
    print(f"Total muestras: {len(data)}")

    X_train, y_train = load_split(data, "train_weak")
    X_val,   y_val   = load_split(data, "val_weak")
    X_test,  y_test  = load_split(data, "gold_eval_pendiente")

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train pos: {sum(y_train)} / {len(y_train)} ({100*sum(y_train)/len(y_train):.1f}%)")
    print(f"Test  pos: {sum(y_test)}  / {len(y_test)}  ({100*sum(y_test)/len(y_test):.1f}%)")

    # Pesos de clase
    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {0: cw[0], 1: cw[1]}
    print("Pesos de clase:", {k: round(v, 3) for k, v in class_weight_dict.items()})

    # 2. Pipeline TF-IDF + LogReg
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=100_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight=class_weight_dict,
            solver="lbfgs",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    print("\nEntrenando TF-IDF + LogReg (binario)...")
    pipeline.fit(X_train, y_train)

    # Val
    val_preds = pipeline.predict(X_val)
    val_f1_pos = f1_score(y_val, val_preds, pos_label=1, zero_division=0)
    val_f1_mac = f1_score(y_val, val_preds, average="macro", zero_division=0)
    print(f"Val  F1-pos: {val_f1_pos:.4f}  |  F1-macro: {val_f1_mac:.4f}")

    # Test
    test_preds = pipeline.predict(X_test)
    test_f1_pos = f1_score(y_test, test_preds, pos_label=1, zero_division=0)
    test_prec   = precision_score(y_test, test_preds, pos_label=1, zero_division=0)
    test_rec    = recall_score(y_test, test_preds, pos_label=1, zero_division=0)
    test_f1_mac = f1_score(y_test, test_preds, average="macro", zero_division=0)

    print(f"\n── Test ──")
    print(classification_report(y_test, test_preds,
                                 target_names=LABEL_NAMES, zero_division=0))

    cm = confusion_matrix(y_test, test_preds, labels=[0, 1]).tolist()

    results = {
        "modelo": "TF-IDF + LogReg (Tarea 2 binario, class-weighted)",
        "tarea": 2,
        "f1_pos":      round(test_f1_pos, 4),
        "precision_pos": round(test_prec, 4),
        "recall_pos":    round(test_rec, 4),
        "f1_macro":    round(test_f1_mac, 4),
        "n_pos_pred":  int(sum(test_preds)),
        "n_pos_true":  int(sum(y_test)),
        "confusion_matrix": {"labels": LABEL_NAMES, "matrix": cm},
        "config": {
            "tfidf_ngram": "(1,2)",
            "tfidf_max_features": 100_000,
            "logreg_C": 1.0,
            "class_weighted_loss": True,
            "train_samples": len(X_train),
            "val_samples":   len(X_val),
            "test_samples":  len(X_test),
            "seed": SEED,
            "dataset": str(DATA_PATH),
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Resultados guardados en {RESULTS_PATH}")
    print(f"  F1-pos  (test): {test_f1_pos:.4f}")
    print(f"  Prec    (test): {test_prec:.4f}")
    print(f"  Recall  (test): {test_rec:.4f}")
    print(f"  F1-macro(test): {test_f1_mac:.4f}")
    print(f"  Pred pos: {int(sum(test_preds))} / True pos: {int(sum(y_test))}")


if __name__ == "__main__":
    main()
