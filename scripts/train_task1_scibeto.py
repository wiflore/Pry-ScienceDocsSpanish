"""
Fine-tuning SciBETO-large para clasificación retórica de 8 clases (Task 1).

Estrategia:
  - Head+Tail truncation (primeros 128 + últimos 382 tokens = 512 total)
  - Split por documento (sin solapamiento entre train/val/test)
  - CrossEntropyLoss con pesos por clase para compensar desbalance
  - Evaluación con Macro F1, reporte por clase y matriz de confusión
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

# ── Configuración ──────────────────────────────────────────────────────────────
SEED = 42
MODEL_ID = "Flaglab/SciBETO-large"
DATA_PATH = Path("data/RawDatasetsV2/DatasetAnotacionManual_Consolidado.xlsx")
OUTPUT_DIR = Path("models/scibeto-task1-8clases")
RESULTS_PATH = Path("reports/entrega3/results_task1_scibeto.json")

# Hiperparámetros
MAX_LENGTH = 512
HEAD_TOKENS = 128      # primeros N tokens del texto
TAIL_TOKENS = 382      # últimos M tokens  (HEAD + TAIL = 510, +CLS+SEP = 512)
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10         # más épocas — early stopping controla cuándo parar
EARLY_STOPPING_PATIENCE = 3  # parar si val_f1 no mejora en N épocas seguidas
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
CLASSIFIER_DROPOUT = 0.3  # default es 0.1, subimos para regularizar
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO implícito = 0.10

LABEL_ORDER = ["BACK", "CONC", "CONTR", "DISC", "INTRO", "LIM", "METH", "RES"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_ORDER)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = len(LABEL_ORDER)


# ── Utilidades ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def head_tail_tokenize(text: str, tokenizer, max_length: int = 512,
                       head: int = 128, tail: int = 382):
    """
    Tokeniza con estrategia Head+Tail.
    Si el texto cabe en max_length, tokeniza normalmente.
    Si no, toma los primeros `head` tokens y los últimos `tail` tokens,
    rodeados por [CLS] y [SEP] (tokens especiales de RoBERTa: <s> y </s>).
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= (max_length - 2):
        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    # Head + Tail
    selected = tokens[:head] + tokens[-tail:]
    # RoBERTa: <s> ... </s>
    input_ids = [tokenizer.bos_token_id] + selected + [tokenizer.eos_token_id]
    seq_len = len(input_ids)
    pad_len = max_length - seq_len
    attention_mask = [1] * seq_len + [0] * pad_len
    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
    )


# ── Dataset ────────────────────────────────────────────────────────────────────
class RhetoricDataset(Dataset):
    def __init__(self, records: list, tokenizer, max_length: int = 512,
                 head: int = 128, tail: int = 382):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head = head
        self.tail = tail

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        input_ids, attention_mask = head_tail_tokenize(
            row["texto"], self.tokenizer,
            self.max_length, self.head, self.tail,
        )
        label = torch.tensor(LABEL2ID[row["label"]], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


# ── Split por documento ─────────────────────────────────────────────────────────
def split_by_document(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                      seed: int = 42):
    """
    Divide los datos asegurando que ningún documento aparezca en más de un split.
    """
    rng = random.Random(seed)
    doc_ids = df["documento_id"].unique().tolist()
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_docs = set(doc_ids[:n_train])
    val_docs = set(doc_ids[n_train: n_train + n_val])
    test_docs = set(doc_ids[n_train + n_val:])

    train_df = df[df["documento_id"].isin(train_docs)].copy()
    val_df = df[df["documento_id"].isin(val_docs)].copy()
    test_df = df[df["documento_id"].isin(test_docs)].copy()

    return train_df, val_df, test_df


def df_to_records(df: pd.DataFrame) -> list:
    return [
        {"texto": row["texto"], "label": row["etiqueta_anotador"]}
        for _, row in df.iterrows()
    ]


# ── Pesos de clase ──────────────────────────────────────────────────────────────
def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Pesos inversamente proporcionales a la frecuencia de cada clase."""
    counts = np.bincount([LABEL2ID[l] for l in labels], minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)  # evitar división por cero
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


# ── Entrenamiento ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Eval ", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()

        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), macro_f1, all_preds, all_labels


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Cargar y limpiar datos
    print("\n[1/5] Cargando dataset...")
    df = pd.read_excel(DATA_PATH)
    df = df[df["etiqueta_anotador"].notna()].copy()
    df = df[df["etiqueta_anotador"].isin(LABEL_ORDER)].copy()
    print(f"  Total muestras: {len(df)}")
    print("  Distribución:")
    print(df["etiqueta_anotador"].value_counts().to_string())

    # 2. Split por documento
    print("\n[2/5] Dividiendo por documento (80/10/10)...")
    train_df, val_df, test_df = split_by_document(df, TRAIN_RATIO, VAL_RATIO, SEED)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("  Test distribución:")
    print(test_df["etiqueta_anotador"].value_counts().to_string())

    # 3. Tokenizador y datasets
    print(f"\n[3/5] Cargando tokenizador ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train_ds = RhetoricDataset(df_to_records(train_df), tokenizer, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS)
    val_ds = RhetoricDataset(df_to_records(val_df), tokenizer, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS)
    test_ds = RhetoricDataset(df_to_records(test_df), tokenizer, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. Modelo
    print(f"\n[4/5] Cargando modelo ({MODEL_ID})...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        classifier_dropout=CLASSIFIER_DROPOUT,
        hidden_dropout_prob=CLASSIFIER_DROPOUT,
    ).to(device)
    model = model.float()

    # Pesos de clase
    class_weights = compute_class_weights(
        train_df["etiqueta_anotador"].tolist(), NUM_LABELS
    ).to(device)
    print("  Pesos de clase:")
    for label, w in zip(LABEL_ORDER, class_weights.cpu().tolist()):
        print(f"    {label}: {w:.3f}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizador y scheduler
    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Entrenamiento
    print(f"\n[5/5] Entrenando (máx {NUM_EPOCHS} épocas, patience={EARLY_STOPPING_PATIENCE})...")
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_model_path = OUTPUT_DIR / "best_model"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nÉpoca {epoch}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, loss_fn, device)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ Mejor modelo guardado (val_f1={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  – Sin mejora ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping en época {epoch}.")
                break

    # Evaluación final en test con el mejor modelo
    print("\n── Evaluación en TEST (mejor modelo) ──")
    model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(device).float()
    _, test_f1, test_preds, test_labels = evaluate(model, test_loader, loss_fn, device)

    pred_names = [ID2LABEL[p] for p in test_preds]
    true_names = [ID2LABEL[l] for l in test_labels]

    report = classification_report(true_names, pred_names, target_names=LABEL_ORDER,
                                   zero_division=0, output_dict=True)
    cm = confusion_matrix(true_names, pred_names, labels=LABEL_ORDER).tolist()

    print(classification_report(true_names, pred_names, target_names=LABEL_ORDER, zero_division=0))

    results = {
        "modelo": "SciBETO-large (fine-tuned, 8 clases, Head+Tail, weighted loss)",
        "macro_f1": round(test_f1, 4),
        "accuracy": round(report["accuracy"], 4),
        "per_class_f1": {l: round(report[l]["f1-score"], 4) for l in LABEL_ORDER},
        "per_class_precision": {l: round(report[l]["precision"], 4) for l in LABEL_ORDER},
        "per_class_recall": {l: round(report[l]["recall"], 4) for l in LABEL_ORDER},
        "per_class_support": {l: int(report[l]["support"]) for l in LABEL_ORDER},
        "confusion_matrix": {"labels": LABEL_ORDER, "matrix": cm},
        "config": {
            "model_id": MODEL_ID,
            "max_length": MAX_LENGTH,
            "head_tokens": HEAD_TOKENS,
            "tail_tokens": TAIL_TOKENS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "seed": SEED,
            "class_weighted_loss": True,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Resultados guardados en {RESULTS_PATH}")
    print(f"  Macro F1 (test): {test_f1:.4f}")


if __name__ == "__main__":
    main()
