"""
Evaluación de Gemini Flash para Tarea 2: extracción de contribuciones científicas
(clasificación binaria: contribucion / no_contribucion).

Modos:
  - zero_shot  : definiciones binarias sin ejemplos
  - few_shot   : definiciones + 1 ejemplo por clase (del train_weak)
  - both       : ambos modos (default)

Uso:
  python scripts/eval_task2_gemini.py
  python scripts/eval_task2_gemini.py --mode zero_shot
  python scripts/eval_task2_gemini.py --mode few_shot

Dataset:
  data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl
  - train_weak (1600) para few-shot sampling
  - gold_eval_pendiente (200, 100/100) para evaluación

Requisitos:
  - Archivo .env con GEMINI_API_KEY=<tu_clave>
  - pip install google-genai python-dotenv scikit-learn tqdm
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
SEED         = 42
DATA_PATH    = Path("data/tarea2/processed/tarea2_dataset_candidatos_2000.jsonl")
RESULTS_DIR  = Path("reports/entrega3")
GEMINI_MODEL = "gemini-2.5-flash"

LABEL_NAMES = ["no_contribucion", "contribucion"]  # índices 0, 1
# Rate limiting: máx 15 req/min en tier gratuito
REQUEST_DELAY_S = 4.2


# ── Prompt ─────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un clasificador experto de fragmentos de artículos científicos en español.

Tu tarea: determinar si el fragmento declara explícitamente una CONTRIBUCIÓN CIENTÍFICA PROPIA del trabajo.

Responde ÚNICAMENTE con una de estas dos etiquetas (sin explicación, sin puntuación extra):
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


# ── Parseo de respuesta ────────────────────────────────────────────────────────
_ALIAS_MAP = {
    "contribucion": 1, "contribución": 1, "contribuciones": 1,
    "contribution": 1, "contributions": 1, "si": 1, "yes": 1, "1": 1,
    "no_contribucion": 0, "no contribucion": 0, "no_contribución": 0,
    "no contribución": 0, "no_contribution": 0, "no": 0, "0": 0,
}


def parse_label(response: str) -> int | None:
    text = response.strip().lower()
    for ch in ".,;:!?\n\t \"'":
        text = text.rstrip(ch)
    # Coincidencia exacta
    if text in _ALIAS_MAP:
        return _ALIAS_MAP[text]
    # Busca primero la etiqueta más larga que coincida
    for alias, label in sorted(_ALIAS_MAP.items(), key=lambda x: -len(x[0])):
        if alias in text:
            return label
    return None


# ── Datos ──────────────────────────────────────────────────────────────────────
def load_data():
    data = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8")]
    train = [d for d in data if d["split"] == "train_weak"]
    test  = [d for d in data if d["split"] == "gold_eval_pendiente"]
    return train, test


def build_fewshot_block(train: list[dict], n_per_class: int = 1,
                        max_words: int = 400) -> str:
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


# ── Cliente Gemini ─────────────────────────────────────────────────────────────
class GeminiClassifier:
    def __init__(self, model: str = GEMINI_MODEL):
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY no encontrada. Añádela a .env")
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def classify(self, text: str, system_prompt: str,
                 few_shot_block: str = "") -> tuple[int | None, float]:
        from google.genai import types
        user_content = USER_TEMPLATE.replace("{text}", text)
        full_prompt = system_prompt
        if few_shot_block:
            full_prompt += "\n" + few_shot_block
        full_prompt += "\n\n" + user_content

        max_retries, wait = 5, 10
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=15,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                latency = time.perf_counter() - t0
                raw = response.text.strip() if response.text else ""
                return parse_label(raw), latency
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "rate" in err:
                    print(f"\n  [rate limit] esperando {wait}s (intento {attempt+1})...")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
                else:
                    print(f"\n  [error] {e}")
                    return None, 0.0
        return None, 0.0


# ── Evaluación ─────────────────────────────────────────────────────────────────
def evaluate(classifier: GeminiClassifier, test: list[dict],
             system_prompt: str, few_shot_block: str = "",
             mode: str = "zero_shot") -> dict:
    y_true, y_pred = [], []
    n_invalid = 0
    latencies = []

    for item in tqdm(test, desc=f"Gemini {mode}"):
        pred, lat = classifier.classify(item["texto"], system_prompt, few_shot_block)
        latencies.append(lat)
        if pred is None:
            n_invalid += 1
            pred = 0  # fallback a negativo
        y_true.append(item["label"])
        y_pred.append(pred)
        time.sleep(REQUEST_DELAY_S)

    f1_pos   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_pos  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_mac   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n── {mode} ──")
    print(classification_report(y_true, y_pred,
                                 target_names=LABEL_NAMES, zero_division=0))

    return {
        "modelo": f"gemini-2.5-flash · Tarea 2 · {mode}",
        "tarea": 2,
        "mode": mode,
        "f1_pos":       round(f1_pos, 4),
        "precision_pos": round(prec_pos, 4),
        "recall_pos":    round(rec_pos, 4),
        "f1_macro":     round(f1_mac, 4),
        "n_pos_pred":   int(sum(y_pred)),
        "n_pos_true":   int(sum(y_true)),
        "n_invalid":    n_invalid,
        "n_test":       len(test),
        "avg_latency_s": round(float(np.mean(latencies)), 3),
        "config": {
            "gemini_model":   GEMINI_MODEL,
            "temperature":    0.0,
            "max_output_tokens": 15,
            "thinking_budget": 0,
            "few_shot_n_per_class": 1 if mode == "few_shot" else 0,
            "few_shot_max_words": 400,
            "request_delay_s": REQUEST_DELAY_S,
            "seed": SEED,
        },
        "y_pred": y_pred,
        "y_true": y_true,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"],
                        default="both")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    train, test = load_data()
    print(f"Train: {len(train)} | Test: {len(test)}")
    print(f"Test pos: {sum(d['label'] for d in test)} / {len(test)}")

    classifier = GeminiClassifier()
    few_shot_block = build_fewshot_block(train, n_per_class=1, max_words=400)

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]

    for mode in modes:
        block = few_shot_block if mode == "few_shot" else ""
        results = evaluate(classifier, test, SYSTEM_PROMPT, block, mode)

        out_path = RESULTS_DIR / f"results_task2_gemini_{mode}.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
