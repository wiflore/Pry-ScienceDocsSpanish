"""
Evaluación de Qwen3 (vía HuggingFace Transformers) para Tarea 2:
extracción de contribuciones científicas (clasificación binaria).

Modos:
  - zero_shot  : definiciones binarias sin ejemplos
  - few_shot   : definiciones + 1 ejemplo por clase (del train_weak)
  - both       : ambos modos (default)

Uso:
  python scripts/eval_task2_qwen.py
  python scripts/eval_task2_qwen.py --mode zero_shot
  python scripts/eval_task2_qwen.py --model Qwen/Qwen3-8B --mode both

Dataset:
  data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl
  - train_weak (1600) para few-shot sampling
  - gold_eval_pendiente (200, 100/100) para evaluación

Requisitos:
  - pip install transformers accelerate
  - GPU: RTX 4070 12 GB (Qwen3-8B en bfloat16, ~5 GB VRAM)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuración ──────────────────────────────────────────────────────────────
SEED          = 42
DATA_PATH     = Path("data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl")
RESULTS_DIR   = Path("reports/entrega3")
DEFAULT_MODEL = "Qwen/Qwen3-8B"

LABEL_NAMES = ["no_contribucion", "contribucion"]  # 0, 1

# Alias para parsear la respuesta del modelo
_ALIAS_MAP = {
    "contribucion": 1, "contribución": 1, "contribuciones": 1,
    "contribution": 1, "contributions": 1, "si": 1, "yes": 1, "1": 1,
    "no_contribucion": 0, "no contribucion": 0, "no_contribución": 0,
    "no contribución": 0, "no_contribution": 0, "no": 0, "0": 0,
}


# ── Utilidades ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label(response: str) -> int | None:
    text = response.strip().lower()
    for ch in ".,;:!?\n\t \"'()[]<>":
        text = text.strip(ch)
    if text in _ALIAS_MAP:
        return _ALIAS_MAP[text]
    for alias in sorted(_ALIAS_MAP.keys(), key=lambda x: -len(x)):
        if alias in text:
            return _ALIAS_MAP[alias]
    return None


# ── Datos ──────────────────────────────────────────────────────────────────────
def load_data():
    data = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]
    train = [d for d in data if d["split"] == "train_weak"]
    test  = [d for d in data if d["split"] == "gold_eval_pendiente"]
    return train, test


def build_fewshot_block(train: list[dict], n_per_class: int = 1,
                        max_words: int = 80) -> str:
    """max_words=80 para evitar overflow de VRAM en GPU local."""
    rng = random.Random(SEED)
    by_label = {0: [], 1: []}
    for d in train:
        by_label[d["label"]].append(d)
    lines = ["\nEjemplos de clasificación:\n"]
    for label, name in [(1, "contribucion"), (0, "no_contribucion")]:
        pool = by_label[label]
        samples = rng.sample(pool, min(n_per_class, len(pool)))
        for s in samples:
            snippet = " ".join(str(s["texto"]).split()[:max_words])
            lines.append(f'Fragmento: "{snippet}"\nEtiqueta: {name}\n')
    return "\n".join(lines)


# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """/no_think
Eres un clasificador experto de fragmentos de artículos científicos en español.

Tu tarea: determinar si el fragmento declara explícitamente una CONTRIBUCIÓN CIENTÍFICA PROPIA del trabajo.

Responde ÚNICAMENTE con una de estas dos etiquetas (sin explicación):
  contribucion
  no_contribucion

Definición de CONTRIBUCIÓN CIENTÍFICA PROPIA:
El fragmento declara que el trabajo propio propone, presenta, desarrolla, introduce,
aporta, construye o valida algo nuevo y propio:
- método, metodología, modelo, sistema, herramienta o algoritmo
- enfoque, marco, arquitectura, estrategia o protocolo
- evidencia, hallazgo, corpus, dataset o recurso propio
- contribución principal, aporte o solución del artículo

La contribución debe estar atribuida al propio trabajo/artículo/estudio.

Marcar como NO CONTRIBUCIÓN cuando:
- describe contexto, antecedentes, marco teórico o resultados sin declarar aporte propio
- menciona contribuciones de OTROS autores
- declara objetivos sin novedad explícita
- resume trabajos previos
- contiene señales como "propone" o "aporte" referidas a terceros"""

USER_TEMPLATE = 'Fragmento: "{text}"\nEtiqueta:'


# ── Clasificador Qwen ──────────────────────────────────────────────────────────
class QwenClassifier:
    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        print(f"  Cargando tokenizador desde {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"  Cargando modelo en {self.device} ({dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        print("  Modelo cargado.")

    def classify(self, text: str, system_prompt: str,
                 few_shot_block: str = "") -> tuple[int | None, float]:
        user_content = USER_TEMPLATE.replace("{text}", text)
        if few_shot_block:
            user_content = few_shot_block.strip() + "\n\n" + user_content

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return parse_label(raw), latency


# ── Evaluación ─────────────────────────────────────────────────────────────────
def evaluate(classifier: QwenClassifier, test: list[dict],
             system_prompt: str, few_shot_block: str,
             mode_name: str) -> dict:
    y_true, y_pred, latencies = [], [], []
    n_invalid = 0

    for item in tqdm(test, desc=f"Qwen [{mode_name}]"):
        pred, lat = classifier.classify(item["texto"], system_prompt, few_shot_block)
        if pred is None:
            n_invalid += 1
            pred = 0  # fallback a negativo
        y_true.append(item["label"])
        y_pred.append(pred)
        latencies.append(lat)

    f1_pos   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_pos  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_mac   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n── {mode_name} ──")
    print(classification_report(y_true, y_pred,
                                 target_names=LABEL_NAMES, zero_division=0))

    lat_arr = np.array(latencies)
    return {
        "modelo": f"{classifier.model_id} · Tarea 2 · {mode_name}",
        "tarea": 2,
        "mode": mode_name,
        "f1_pos":        round(f1_pos, 4),
        "precision_pos": round(prec_pos, 4),
        "recall_pos":    round(rec_pos, 4),
        "f1_macro":      round(f1_mac, 4),
        "n_pos_pred":    int(sum(y_pred)),
        "n_pos_true":    int(sum(y_true)),
        "n_invalid":     n_invalid,
        "n_test":        len(test),
        "latency": {
            "total_s":  round(float(lat_arr.sum()), 2),
            "mean_s":   round(float(lat_arr.mean()), 3),
            "p95_s":    round(float(np.percentile(lat_arr, 95)), 3),
        },
        "config": {
            "model":           classifier.model_id,
            "thinking":        False,
            "max_new_tokens":  15,
            "few_shot_n_per_class": 1 if "few" in mode_name else 0,
            "few_shot_max_words": 80,
            "seed": SEED,
        },
        "y_pred": y_pred,
        "y_true": y_true,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", default="both",
                        choices=["zero_shot", "few_shot", "both"])
    args = parser.parse_args()

    set_seed(SEED)
    print(f"Modelo: {args.model}")
    dev = f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    print("\n[1/3] Cargando dataset Tarea 2...")
    train, test = load_data()
    print(f"  Train: {len(train)} | Test: {len(test)}")
    print(f"  Test pos: {sum(d['label'] for d in test)} / {len(test)}")

    print("[2/3] Cargando modelo Qwen3...")
    classifier = QwenClassifier(model_id=args.model)

    few_shot_block = build_fewshot_block(train, n_per_class=1, max_words=80)
    model_slug = args.model.replace("/", "_").replace(":", "_")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]
    print(f"[3/3] Evaluando modos: {modes}")

    for mode in modes:
        fs_block = few_shot_block if mode == "few_shot" else ""
        results = evaluate(classifier, test, SYSTEM_PROMPT, fs_block, mode)

        out_path = RESULTS_DIR / f"results_task2_{model_slug}_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Guardado en {out_path}")
        print(f"  F1-pos:   {results['f1_pos']:.4f}")
        print(f"  Prec:     {results['precision_pos']:.4f}")
        print(f"  Recall:   {results['recall_pos']:.4f}")
        print(f"  F1-macro: {results['f1_macro']:.4f}")
        print(f"  Pred pos: {results['n_pos_pred']} / True pos: {results['n_pos_true']}")
        print(f"  Inválidos:{results['n_invalid']}")


if __name__ == "__main__":
    main()
