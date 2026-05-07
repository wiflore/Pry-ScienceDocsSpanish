"""Fine-tuning de SciBETO-large para clasificación retórica IMRaD."""

from pathlib import Path

import torch
from datasets import DatasetDict
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

MODEL_HF_ID = "Flaglab/SciBETO-large"
NUM_ETIQUETAS = 4
ETIQUETAS = ["Introducción", "Metodología", "Resultados", "Discusión"]


class SciBETOClassifier:
    """Entrena y clasifica con SciBETO-large."""

    def __init__(
        self,
        device,
        model_id=MODEL_HF_ID,
        max_length=512,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        output_dir="models/scibeto-large-imrad",
        num_labels: int = NUM_ETIQUETAS,
        label_column: str = "imrad_label",
        truncation_strategy: str = "standard",  # "standard" | "head_tail"
        class_weights=None,  # tensor [num_labels] o None
    ):
        self.device = device
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.output_dir = Path(output_dir)
        self.num_labels = num_labels
        self.label_column = label_column
        self.truncation_strategy = truncation_strategy
        if class_weights is not None and not isinstance(class_weights, torch.Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.class_weights = (
            class_weights.to(device) if class_weights is not None else None
        )

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        ).to(device)
        # float32 requerido para MPS (Apple Silicon)
        self.model = self.model.float()

    def _encode_head_tail(self, textos):
        """Tokeniza con estrategia head+tail: primeros H + últimos T tokens.

        Reserva 2 tokens para [CLS]/[SEP]. Por defecto H=T=(max_length-2)//2.
        """
        max_content = self.max_length - 2
        head_len = max_content // 2
        tail_len = max_content - head_len
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        input_ids_batch = []
        attention_batch = []
        for texto in textos:
            ids = self.tokenizer(texto, add_special_tokens=False, truncation=False)["input_ids"]
            if len(ids) <= max_content:
                kept = ids
            else:
                kept = ids[:head_len] + ids[-tail_len:]
            seq = [cls_id] + kept + [sep_id]
            attn = [1] * len(seq)
            # Padding hasta max_length
            pad_n = self.max_length - len(seq)
            if pad_n > 0:
                seq = seq + [pad_id] * pad_n
                attn = attn + [0] * pad_n
            input_ids_batch.append(seq)
            attention_batch.append(attn)
        return {"input_ids": input_ids_batch, "attention_mask": attention_batch}

    def _tokenizar(self, splits: DatasetDict, columna_texto="text_es"):
        """Tokeniza los splits y prepara columnas para PyTorch."""
        if self.truncation_strategy == "head_tail":
            def tokenize_fn(ejemplos):
                return self._encode_head_tail(ejemplos[columna_texto])
        else:
            def tokenize_fn(ejemplos):
                return self.tokenizer(
                    ejemplos[columna_texto],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                )

        tokenized = splits.map(tokenize_fn, batched=True, desc="Tokenizando")
        # Renombrar columna de etiquetas al nombre que espera el modelo
        if "labels" in tokenized["train"].column_names and self.label_column != "labels":
            tokenized = tokenized.remove_columns("labels")
        if self.label_column != "labels":
            tokenized = tokenized.rename_column(self.label_column, "labels")
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized

    def train(self, splits: DatasetDict, columna_texto="text_es"):
        """Entrena sobre splits['train'] y valida en splits['validation']."""
        tokenized = self._tokenizar(splits, columna_texto)

        train_loader = DataLoader(tokenized["train"], batch_size=self.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(tokenized["validation"], batch_size=self.batch_size,
                                shuffle=False, num_workers=0)

        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        mejor_acc = 0.0
        loss_fn = (
            torch.nn.CrossEntropyLoss(weight=self.class_weights)
            if self.class_weights is not None
            else None
        )
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            loss_total = 0.0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                if loss_fn is not None:
                    loss = loss_fn(outputs.logits, batch["labels"])
                else:
                    loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                loss_total += loss.item()

            val_metrics = self._evaluar(val_loader)
            print(f"Epoch {epoch} — loss: {loss_total/len(train_loader):.4f} | "
                  f"val_acc: {val_metrics['accuracy']:.4f}")

            if val_metrics["accuracy"] > mejor_acc:
                mejor_acc = val_metrics["accuracy"]
                self.save(self.output_dir)

        print(f"Mejor val_acc: {mejor_acc:.4f} — modelo guardado en {self.output_dir}")
        return val_metrics

    def _evaluar(self, loader: DataLoader):
        """Calcula accuracy sobre un DataLoader."""
        self.model.eval()
        correctos = total = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.model(**batch).logits.argmax(dim=-1)
                correctos += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])
        return {"accuracy": correctos / total if total > 0 else 0.0}

    def predict(self, textos: list) -> list:
        """Predice etiquetas IMRaD para una lista de textos."""
        self.model.eval()
        if self.truncation_strategy == "head_tail":
            enc = self._encode_head_tail(textos)
            inputs = {
                "input_ids": torch.tensor(enc["input_ids"]).to(self.device),
                "attention_mask": torch.tensor(enc["attention_mask"]).to(self.device),
            }
        else:
            inputs = self.tokenizer(
                textos, return_tensors="pt", truncation=True,
                padding=True, max_length=self.max_length,
            ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.argmax(dim=-1).cpu().tolist()

    def save(self, path):
        """Guarda modelo y tokenizer en disco."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))

    @classmethod
    def load(cls, path, device):
        """Carga un modelo previamente guardado."""
        path = Path(path)
        inst = cls.__new__(cls)
        inst.device = device
        inst.max_length = 512
        inst.tokenizer = RobertaTokenizerFast.from_pretrained(str(path))
        inst.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(device)
        inst.model = inst.model.float()
        inst.output_dir = path
        inst.num_labels = inst.model.config.num_labels
        inst.label_column = "labels"
        inst.truncation_strategy = "standard"
        return inst
