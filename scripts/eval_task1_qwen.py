"""
Evaluación de Qwen3 (vía HuggingFace transformers) para clasificación retórica
de 8 clases (Tarea 1).

Usa inferencia directa con AutoModelForCausalLM — sin Ollama, sin dependencias
adicionales. Compatible con el mismo entorno PyTorch+CUDA usado para SciBETO.
Apto para deploy en AWS (mismo Dockerfile que el resto del backend).

Modos:
  - zero_shot  : definiciones de las 8 clases, sin ejemplos
  - few_shot   : definiciones + 1 ejemplo por clase (sampled del train set, seed=42)
  - both       : ejecuta ambos (default)

Uso:
  python scripts/eval_task1_qwen.py
  python scripts/eval_task1_qwen.py --model Qwen/Qwen3-8B --mode zero_shot
  python scripts/eval_task1_qwen.py --model Qwen/Qwen3-1.7B --mode both

Requisitos:
  - pip install transformers accelerate
  - GPU recomendada (RTX 4070 12 GB soporta Qwen3-8B en bfloat16, ~5 GB VRAM)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Configuración ──────────────────────────────────────────────────────────────
SEED = 42
DATA_PATH = Path("data/RawDatasetsV2/DatasetAnotacionManual_Consolidado.xlsx")
RESULTS_DIR = Path("reports/entrega3")
DEFAULT_MODEL = "Qwen/Qwen3-8B"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
LABEL_ORDER = ["BACK", "CONC", "CONTR", "DISC", "INTRO", "LIM", "METH", "RES"]

# Alias para parsear respuesta (minúsculas → código canónico)
ALIAS_MAP = {
    # Códigos directos
    "back": "BACK", "conc": "CONC", "contr": "CONTR", "disc": "DISC",
    "intro": "INTRO", "lim": "LIM", "meth": "METH", "res": "RES",
    # Nombres en español
    "antecedentes": "BACK", "antecedente": "BACK", "estado del arte": "BACK",
    "conclusiones": "CONC", "conclusión": "CONC", "conclusion": "CONC",
    "contribuciones": "CONTR", "contribución": "CONTR", "contribution": "CONTR",
    "discusión": "DISC", "discusion": "DISC", "discussion": "DISC",
    "introducción": "INTRO", "introduccion": "INTRO", "introduction": "INTRO",
    "limitaciones": "LIM", "limitación": "LIM", "limitation": "LIM", "limitations": "LIM",
    "metodología": "METH", "metodologia": "METH", "methodology": "METH", "methods": "METH",
    "resultados": "RES", "results": "RES", "resultado": "RES",
    # Variantes en inglés
    "background": "BACK", "conclusions": "CONC", "contributions": "CONTR",
    "method": "METH", "result": "RES",
}


# ── Utilidades ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_label(response: str) -> str | None:
    """Extrae la etiqueta canónica de la respuesta del modelo."""
    text = response.strip().lower()
    for ch in ".,;:!?\n\t \"'()[]<>":
        text = text.strip(ch)
    if text in ALIAS_MAP:
        return ALIAS_MAP[text]
    # Buscar la mención más larga conocida dentro del texto
    for alias in sorted(ALIAS_MAP.keys(), key=lambda x: -len(x)):
        if alias in text:
            return ALIAS_MAP[alias]
    return None


def split_by_document(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                      seed: int = 42):
    """División idéntica a la usada en SciBETO y Gemini."""
    rng = random.Random(seed)
    doc_ids = df["documento_id"].unique().tolist()
    rng.shuffle(doc_ids)
    n = len(doc_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_docs = set(doc_ids[:n_train])
    val_docs = set(doc_ids[n_train: n_train + n_val])
    test_docs = set(doc_ids[n_train + n_val:])
    return (
        df[df["documento_id"].isin(train_docs)].copy(),
        df[df["documento_id"].isin(val_docs)].copy(),
        df[df["documento_id"].isin(test_docs)].copy(),
    )


# ── Prompts ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un clasificador experto de fragmentos de artículos científicos en español.

Tu tarea: asignar UNA etiqueta al fragmento según su función retórica dentro del documento.

Responde ÚNICAMENTE con el código de la etiqueta, sin explicación, sin puntuación, sin texto adicional.

Etiquetas válidas y sus definiciones:
- INTRO  : Presenta el problema de investigación, motivación, objetivos y, a veces, una descripción general del enfoque propuesto.
- BACK   : Describe el estado del arte, trabajos previos relevantes y el contexto teórico en que se enmarca la investigación.
- METH   : Explica el diseño experimental, métodos, modelos, datos, materiales y procedimientos utilizados.
- RES    : Presenta los resultados obtenidos (cifras, tablas, evaluaciones) generalmente sin interpretación extensiva.
- DISC   : Interpreta los resultados, analiza sus implicaciones y los compara con trabajos previos.
- CONTR  : Identifica explícitamente los aportes originales del trabajo (métodos propuestos, hallazgos principales, avances conceptuales).
- LIM    : Describe restricciones del enfoque, supuestos adoptados, posibles fuentes de error o límites de generalización.
- CONC   : Resume los principales hallazgos del trabajo y presenta líneas de trabajo futuro.

Reglas de desambiguación (en orden de prioridad):
1. Si el texto lista contribuciones originales explícitas ("este trabajo propone", "se presenta un nuevo método", "la principal contribución es"), usa CONTR.
2. Si el texto menciona restricciones, supuestos o limitaciones del propio enfoque, usa LIM.
3. Si el texto resume hallazgos y menciona trabajo futuro, usa CONC.
4. Si el texto describe procedimientos, datos o diseño experimental, usa METH.
5. Si el texto cita y describe trabajos previos sin reportar los propios resultados, usa BACK.
6. Si el texto interpreta o discute los propios resultados, usa DISC.
7. Si el texto reporta datos o medidas sin interpretación, usa RES.
8. Si el texto presenta el problema, la brecha de conocimiento o los objetivos, usa INTRO."""

USER_TEMPLATE = 'Fragmento: "{text}"\nEtiqueta:'


def build_fewshot_block(train_df: pd.DataFrame, rng: random.Random,
                        n_per_class: int = 1) -> str:
    """Construye el bloque few-shot con n_per_class ejemplos por clase."""
    lines = ["\nEjemplos de clasificación:\n"]
    for label in LABEL_ORDER:
        subset = train_df[train_df["etiqueta_anotador"] == label]
        samples = subset.sample(
            n=min(n_per_class, len(subset)),
            random_state=rng.randint(0, 9999),
        )
        for _, row in samples.iterrows():
            # 80 palabras: suficiente para dar el patrón sin saturar la VRAM
            snippet = " ".join(str(row["texto"]).split()[:80])
            lines.append(f'Fragmento: "{snippet}"\nEtiqueta: {label}\n')
    return "\n".join(lines)


# ── Clasificador Qwen con transformers ────────────────────────────────────────
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
            device_map="auto",   # distribuye en GPU/CPU automáticamente
        )
        self.model.eval()
        print(f"  Modelo cargado.")

    def classify(self, text: str, system_prompt: str,
                 few_shot_block: str = "") -> tuple[str | None, float]:
        """Clasifica un fragmento. Devuelve (etiqueta_parseada, latencia_s)."""
        user_content = USER_TEMPLATE.replace("{text}", text)
        if few_shot_block:
            user_content = few_shot_block.strip() + "\n\n" + user_content

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # apply_chat_template formatea el prompt según el modelo
        # enable_thinking=False desactiva chain-of-thought en Qwen3
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
                do_sample=False,          # greedy decoding → determinista
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0

        # Decodificar solo los tokens nuevos (excluir el prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return parse_label(raw), latency


# ── Evaluación ─────────────────────────────────────────────────────────────────
def evaluate(classifier: QwenClassifier, test_df: pd.DataFrame,
             system_prompt: str, few_shot_block: str,
             mode_name: str) -> dict:
    texts = test_df["texto"].tolist()
    true_labels = test_df["etiqueta_anotador"].tolist()

    pred_labels = []
    latencies = []
    unparsed = 0

    for text in tqdm(texts, desc=f"Qwen [{mode_name}]"):
        label, lat = classifier.classify(text, system_prompt, few_shot_block)
        if label is None:
            label = "BACK"  # fallback a clase mayoritaria
            unparsed += 1
        pred_labels.append(label)
        latencies.append(lat)

    macro_f1 = f1_score(true_labels, pred_labels, labels=LABEL_ORDER,
                        average="macro", zero_division=0)
    report = classification_report(
        true_labels, pred_labels, labels=LABEL_ORDER,
        zero_division=0, output_dict=True,
    )
    report_str = classification_report(
        true_labels, pred_labels, labels=LABEL_ORDER,
        zero_division=0,
    )

    lat_arr = np.array(latencies)
    latency_stats = {
        "total_s": round(float(lat_arr.sum()), 2),
        "mean_s": round(float(lat_arr.mean()), 3),
        "median_s": round(float(np.median(lat_arr)), 3),
        "p95_s": round(float(np.percentile(lat_arr, 95)), 3),
        "samples": len(latencies),
    }

    print(f"\n── Evaluación en TEST [{mode_name}] ──")
    print(report_str)
    print(f"Macro F1 (test): {macro_f1:.4f}")
    print(f"Respuestas no parseadas: {unparsed}/{len(texts)}")
    print(f"Latencia media: {latency_stats['mean_s']:.2f}s | total: {latency_stats['total_s']:.0f}s")

    return {
        "mode": mode_name,
        "model": classifier.model_id,
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(report["accuracy"], 4),
        "classification_report": report,
        "unparsed_count": unparsed,
        "n_test": len(texts),
        "latency": latency_stats,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Modelo HuggingFace (ej: Qwen/Qwen3-8B, Qwen/Qwen3-1.7B)")
    parser.add_argument("--mode", default="both",
                        choices=["zero_shot", "few_shot", "both"])
    args = parser.parse_args()

    set_seed(SEED)
    rng = random.Random(SEED)

    print(f"Modelo: {args.model}")
    print(f"Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")
    print("\n[1/4] Cargando dataset...")
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=["etiqueta_anotador"]).reset_index(drop=True)
    print(f"  Total muestras: {len(df)}")

    print("[2/4] Dividiendo por documento (80/10/10)...")
    train_df, _, test_df = split_by_document(df, TRAIN_RATIO, VAL_RATIO, seed=SEED)
    print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

    print("[3/4] Cargando modelo Qwen3...")
    classifier = QwenClassifier(model_id=args.model)

    few_shot_block = build_fewshot_block(train_df, rng, n_per_class=1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]

    # Nombre limpio para el archivo (Qwen/Qwen3-8B → Qwen_Qwen3-8B)
    model_slug = args.model.replace("/", "_").replace(":", "_")

    print(f"[4/4] Evaluando modos: {modes}")
    for mode in modes:
        fs_block = few_shot_block if mode == "few_shot" else ""
        results = evaluate(
            classifier, test_df,
            system_prompt=SYSTEM_PROMPT,
            few_shot_block=fs_block,
            mode_name=mode,
        )
        out_path = RESULTS_DIR / f"results_task1_{model_slug}_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Resultados guardados en {out_path}")

    print("\n✓ Evaluación completa.")


if __name__ == "__main__":
    main()
