"""
Fine-tuning SciBETO-large para Tarea 2: extracción de contribuciones científicas
(clasificación binaria: no_contribucion=0, contribucion=1).

Estrategia:
  - Head+Tail truncation (128 + 382 = 510 + CLS/SEP = 512)
  - Split: train_weak (1600) / val_weak (200) / gold_eval_pendiente (200)
  - CrossEntropyLoss con pesos por clase
  - Early stopping por F1-pos (F1 de la clase positiva)
  - Métrica principal: F1 de la clase positiva (label=1, contribucion)
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ── Configuración ──────────────────────────────────────────────────────────────
SEED       = 42
MODEL_ID   = "Flaglab/SciBETO-large"
DATA_PATH  = Path("data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl")
OUTPUT_DIR = Path("models/scibeto-task2-binario")
RESULTS_PATH = Path("reports/entrega3/results_task2_scibeto.json")

# Hiperparámetros (idénticos a Task 1 para comparabilidad)
MAX_LENGTH  = 512
HEAD_TOKENS = 128
TAIL_TOKENS = 382
BATCH_SIZE  = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS  = 10
EARLY_STOPPING_PATIENCE = 3
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
CLASSIFIER_DROPOUT = 0.3

LABEL_NAMES = ["no_contribucion", "contribucion"]  # índices 0, 1
NUM_LABELS  = 2


# ── Utilidades ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def head_tail_tokenize(text: str, tokenizer, max_length: int = 512,
                       head: int = 128, tail: int = 382):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= (max_length - 2):
        enc = tokenizer(text, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    selected = tokens[:head] + tokens[-tail:]
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
class BinaryDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer,
                 max_length: int = 512, head: int = 128, tail: int = 382):
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
            row["texto"], self.tokenizer, self.max_length, self.head, self.tail)
        label = torch.tensor(row["label"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": label}


# ── Datos ──────────────────────────────────────────────────────────────────────
def load_split(data: list[dict], split_name: str) -> list[dict]:
    return [d for d in data if d["split"] == split_name]


# ── Pesos de clase ─────────────────────────────────────────────────────────────
def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float)


# ── Entrenamiento ──────────────────────────────────────────────────────────────
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

    f1_pos = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return total_loss / len(loader), f1_pos, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Cargar datos
    print("\n[1/5] Cargando dataset Tarea 2...")
    all_data = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]
    train_data = load_split(all_data, "train_weak")
    val_data   = load_split(all_data, "val_weak")
    test_data  = load_split(all_data, "gold_eval_pendiente")
    print(f"  Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"  Train pos: {sum(d['label'] for d in train_data)} / {len(train_data)}")
    print(f"  Test  pos: {sum(d['label'] for d in test_data)}  / {len(test_data)}")

    # 2. Tokenizador y datasets
    print(f"\n[2/5] Cargando tokenizador ({MODEL_ID})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    train_ds = BinaryDataset(train_data, tokenizer, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS)
    val_ds   = BinaryDataset(val_data,   tokenizer, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS)
    test_ds  = BinaryDataset(test_data,  tokenizer, MAX_LENGTH, HEAD_TOKENS, TAIL_TOKENS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Modelo
    print(f"\n[3/5] Cargando modelo ({MODEL_ID})...")
    id2label = {i: l for i, l in enumerate(LABEL_NAMES)}
    label2id = {l: i for i, l in enumerate(LABEL_NAMES)}
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        classifier_dropout=CLASSIFIER_DROPOUT,
        hidden_dropout_prob=CLASSIFIER_DROPOUT,
    ).to(device).float()

    # Pesos de clase
    train_labels = [d["label"] for d in train_data]
    class_weights = compute_class_weights(train_labels, NUM_LABELS).to(device)
    print("  Pesos de clase:")
    for name, w in zip(LABEL_NAMES, class_weights.cpu().tolist()):
        print(f"    {name}: {w:.3f}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Optimizador y scheduler
    num_training_steps = NUM_EPOCHS * len(train_loader)
    num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Entrenamiento con early stopping por F1-pos
    print(f"\n[4/5] Entrenando (máx {NUM_EPOCHS} épocas, patience={EARLY_STOPPING_PATIENCE})...")
    best_val_f1 = -1.0
    epochs_no_improve = 0
    best_model_path = OUTPUT_DIR / "best_model"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nÉpoca {epoch}/{NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, loss_fn, device)
        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_f1_pos={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ Mejor modelo guardado (val_f1_pos={val_f1:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  – Sin mejora ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping en época {epoch}.")
                break

    # 6. Evaluación final en test
    print("\n[5/5] Evaluación en TEST (mejor modelo)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path).to(device).float()
    _, test_f1_pos, test_preds, test_labels_raw = evaluate(
        model, test_loader, loss_fn, device)

    test_prec = precision_score(test_labels_raw, test_preds, pos_label=1, zero_division=0)
    test_rec  = recall_score(test_labels_raw, test_preds, pos_label=1, zero_division=0)
    test_f1_mac = f1_score(test_labels_raw, test_preds, average="macro", zero_division=0)

    report = classification_report(test_labels_raw, test_preds,
                                    target_names=LABEL_NAMES, zero_division=0,
                                    output_dict=True)
    cm = confusion_matrix(test_labels_raw, test_preds, labels=[0, 1]).tolist()
    print(classification_report(test_labels_raw, test_preds,
                                 target_names=LABEL_NAMES, zero_division=0))

    results = {
        "modelo": "SciBETO-large (fine-tuned, Tarea 2 binario, Head+Tail, weighted loss)",
        "tarea": 2,
        "f1_pos":        round(test_f1_pos, 4),
        "precision_pos": round(test_prec, 4),
        "recall_pos":    round(test_rec, 4),
        "f1_macro":      round(test_f1_mac, 4),
        "n_pos_pred":    int(sum(test_preds)),
        "n_pos_true":    int(sum(test_labels_raw)),
        "confusion_matrix": {"labels": LABEL_NAMES, "matrix": cm},
        "config": {
            "model_id":   MODEL_ID,
            "max_length": MAX_LENGTH,
            "head_tokens": HEAD_TOKENS,
            "tail_tokens": TAIL_TOKENS,
            "batch_size":  BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs":  NUM_EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "classifier_dropout": CLASSIFIER_DROPOUT,
            "seed": SEED,
            "class_weighted_loss": True,
            "train_samples": len(train_data),
            "val_samples":   len(val_data),
            "test_samples":  len(test_data),
            "dataset": str(DATA_PATH),
        },
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Resultados guardados en {RESULTS_PATH}")
    print(f"  F1-pos:   {test_f1_pos:.4f}")
    print(f"  Prec:     {test_prec:.4f}")
    print(f"  Recall:   {test_rec:.4f}")
    print(f"  F1-macro: {test_f1_mac:.4f}")
    print(f"  Pred pos: {sum(test_preds)} / True pos: {sum(test_labels_raw)}")


if __name__ == "__main__":
    main()
