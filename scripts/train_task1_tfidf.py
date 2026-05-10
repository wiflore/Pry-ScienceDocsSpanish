"""
Baseline TF-IDF + Logistic Regression para clasificación retórica de 8 clases.
Mismo split por documento que SciBETO para comparación directa.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = 42
DATA_PATH = Path("data/RawDatasetsV2/DatasetAnotacionManual_Consolidado.xlsx")
RESULTS_PATH = Path("reports/entrega3/results_task1_tfidf.json")
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10

LABEL_ORDER = ["BACK", "CONC", "CONTR", "DISC", "INTRO", "LIM", "METH", "RES"]


def split_by_document(df, train_ratio, val_ratio, seed=42):
    rng = random.Random(seed)
    doc_ids = df["documento_id"].unique().tolist()
    rng.shuffle(doc_ids)
    n = len(doc_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_docs = set(doc_ids[:n_train])
    val_docs   = set(doc_ids[n_train: n_train + n_val])
    test_docs  = set(doc_ids[n_train + n_val:])
    return (
        df[df["documento_id"].isin(train_docs)].copy(),
        df[df["documento_id"].isin(val_docs)].copy(),
        df[df["documento_id"].isin(test_docs)].copy(),
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # 1. Datos
    df = pd.read_excel(DATA_PATH)
    df = df[df["etiqueta_anotador"].notna()].copy()
    df = df[df["etiqueta_anotador"].isin(LABEL_ORDER)].copy()
    print(f"Total muestras: {len(df)}")

    train_df, val_df, test_df = split_by_document(df, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    X_train = train_df["texto"].tolist()
    y_train = train_df["etiqueta_anotador"].tolist()
    X_val   = val_df["texto"].tolist()
    y_val   = val_df["etiqueta_anotador"].tolist()
    X_test  = test_df["texto"].tolist()
    y_test  = test_df["etiqueta_anotador"].tolist()

    # Pesos de clase
    classes = np.array(LABEL_ORDER)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(LABEL_ORDER, cw))
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
            multi_class="multinomial",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])

    print("\nEntrenando TF-IDF + LogReg...")
    pipeline.fit(X_train, y_train)

    # Val
    val_preds = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    print(f"Val Macro F1: {val_f1:.4f}")

    # Test
    test_preds = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    print(f"\n── Test ──")
    print(classification_report(y_test, test_preds, target_names=LABEL_ORDER, zero_division=0))

    report = classification_report(y_test, test_preds, target_names=LABEL_ORDER,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, test_preds, labels=LABEL_ORDER).tolist()

    results = {
        "modelo": "TF-IDF + LogReg (baseline, 8 clases, class-weighted)",
        "macro_f1": round(test_f1, 4),
        "accuracy": round(report["accuracy"], 4),
        "per_class_f1":        {l: round(report[l]["f1-score"],  4) for l in LABEL_ORDER},
        "per_class_precision":  {l: round(report[l]["precision"], 4) for l in LABEL_ORDER},
        "per_class_recall":     {l: round(report[l]["recall"],    4) for l in LABEL_ORDER},
        "per_class_support":    {l: int(report[l]["support"])         for l in LABEL_ORDER},
        "confusion_matrix": {"labels": LABEL_ORDER, "matrix": cm},
        "config": {
            "tfidf_ngram": "(1,2)",
            "tfidf_max_features": 100_000,
            "logreg_C": 1.0,
            "class_weighted_loss": True,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "seed": SEED,
        },
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Resultados guardados en {RESULTS_PATH}")
    print(f"  Macro F1 (test): {test_f1:.4f}")


if __name__ == "__main__":
    main()
