"""
Evaluación de Gemini Flash para clasificación retórica de 8 clases (Tarea 1).

Modos:
  - zero_shot  : definiciones de las 8 clases, sin ejemplos
  - few_shot   : definiciones + 1 ejemplo por clase (sampled del train set, seed=42)

Uso:
  python scripts/eval_task1_gemini.py
  python scripts/eval_task1_gemini.py --mode zero_shot
  python scripts/eval_task1_gemini.py --mode few_shot
  python scripts/eval_task1_gemini.py --mode both     (default)

Requisitos:
  - Archivo .env con GEMINI_API_KEY=<tu_clave>
  - pip install google-genai python-dotenv pandas scikit-learn openpyxl tqdm
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

load_dotenv()

import os

# ── Configuración ──────────────────────────────────────────────────────────────
SEED = 42
DATA_PATH = Path("data/RawDatasetsV2/DatasetAnotacionManual_Consolidado.xlsx")
RESULTS_DIR = Path("reports/entrega3")
GEMINI_MODEL = "gemini-2.5-flash"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
LABEL_ORDER = ["BACK", "CONC", "CONTR", "DISC", "INTRO", "LIM", "METH", "RES"]

# Alias para parsear la respuesta del modelo (minúsculas → código canónico)
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
    # Variantes abreviadas en inglés que Gemini suele usar
    "background": "BACK", "conclusion": "CONC", "contributions": "CONTR",
    "limitations": "LIM", "method": "METH", "result": "RES",
}

# Rate limiting: máx 15 req/min en el tier gratuito de Gemini Flash
REQUEST_DELAY_S = 4.2  # ~14 req/min para dejar margen


# ── Utilidades ─────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def parse_label(response: str) -> str | None:
    """Extrae la etiqueta canónica de la respuesta del modelo."""
    text = response.strip().lower()
    # Limpiar puntuación al final
    for ch in ".,;:!?\n\t \"'":
        text = text.rstrip(ch)
    # Coincidencia exacta
    if text in ALIAS_MAP:
        return ALIAS_MAP[text]
    # Buscar la primera mención de una etiqueta conocida
    for alias, label in sorted(ALIAS_MAP.items(), key=lambda x: -len(x[0])):
        if alias in text:
            return label
    return None


def split_by_document(df: pd.DataFrame, train_ratio: float, val_ratio: float,
                      seed: int = 42):
    """Divide por documento, sin solapamiento entre splits (idéntico a SciBETO)."""
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


# ── Construcción de prompts ────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un clasificador experto de fragmentos de artículos científicos en español.

Tu tarea: asignar UNA etiqueta al fragmento según su función retórica dentro del documento.

Responde ÚNICAMENTE con el código de la etiqueta (sin explicación, sin puntuación, sin texto adicional).

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


def build_fewshot_examples(train_df: pd.DataFrame, rng: random.Random,
                           n_per_class: int = 1) -> str:
    """Selecciona n_per_class ejemplos por clase del train set y construye el bloque few-shot."""
    lines = ["\nEjemplos de clasificación:\n"]
    for label in LABEL_ORDER:
        subset = train_df[train_df["etiqueta_anotador"] == label]
        samples = subset.sample(n=min(n_per_class, len(subset)),
                                random_state=rng.randint(0, 9999))
        for _, row in samples.iterrows():
            # Truncar a 400 palabras para no saturar el contexto
            snippet = " ".join(str(row["texto"]).split()[:400])
            lines.append(f'Fragmento: "{snippet}"\nEtiqueta: {label}\n')
    return "\n".join(lines)


# ── Cliente Gemini ─────────────────────────────────────────────────────────────
class GeminiClassifier:
    def __init__(self, model: str = GEMINI_MODEL):
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "TU_NUEVA_CLAVE_AQUI":
            raise ValueError(
                "GEMINI_API_KEY no encontrada. "
                "Edita el archivo .env con tu clave y vuelve a ejecutar."
            )
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def classify(self, text: str, system_prompt: str, few_shot_block: str = "") -> tuple[str | None, float]:
        """Clasifica un fragmento. Devuelve (etiqueta_parseada, latencia_s)."""
        from google.genai import types

        user_content = USER_TEMPLATE.replace("{text}", text)
        full_prompt = system_prompt
        if few_shot_block:
            full_prompt += "\n" + few_shot_block
        full_prompt += "\n\n" + user_content

        max_retries = 5
        wait = 10  # segundos entre reintentos
        for attempt in range(max_retries):
            try:
                t0 = time.perf_counter()
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=10,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                latency = time.perf_counter() - t0
                raw = response.text.strip() if response.text else ""
                return parse_label(raw), latency
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate" in err_str:
                    print(f"\n  [rate limit] esperando {wait}s (intento {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
                else:
                    print(f"\n  [error API] {e}")
                    return None, 0.0
        return None, 0.0


# ── Evaluación ─────────────────────────────────────────────────────────────────
def evaluate(classifier: GeminiClassifier, test_df: pd.DataFrame,
             system_prompt: str, few_shot_block: str,
             mode_name: str) -> dict:
    """Evalúa el clasificador sobre el test set y devuelve métricas."""
    texts = test_df["texto"].tolist()
    true_labels = test_df["etiqueta_anotador"].tolist()

    pred_labels = []
    latencies = []
    unparsed = 0

    for text in tqdm(texts, desc=f"Gemini [{mode_name}]"):
        label, lat = classifier.classify(text, system_prompt, few_shot_block)
        if label is None:
            label = "BACK"  # fallback a clase mayoritaria
            unparsed += 1
        pred_labels.append(label)
        latencies.append(lat)
        time.sleep(REQUEST_DELAY_S)

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
        "model": GEMINI_MODEL,
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
    parser.add_argument("--mode", default="both",
                        choices=["zero_shot", "few_shot", "both"])
    parser.add_argument("--model", default=GEMINI_MODEL,
                        help="Modelo Gemini a usar (ej: gemini-2.0-flash)")
    args = parser.parse_args()

    set_seed(SEED)
    rng = random.Random(SEED)

    print("[1/4] Cargando dataset...")
    df = pd.read_excel(DATA_PATH)
    df = df.dropna(subset=["etiqueta_anotador"]).reset_index(drop=True)
    print(f"  Total muestras: {len(df)}")

    print("[2/4] Dividiendo por documento (80/10/10)...")
    train_df, val_df, test_df = split_by_document(df, TRAIN_RATIO, VAL_RATIO, seed=SEED)
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    print("[3/4] Inicializando cliente Gemini...")
    classifier = GeminiClassifier(model=args.model)

    # Bloque few-shot (se construye una sola vez)
    few_shot_block = build_fewshot_examples(train_df, rng, n_per_class=1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]

    print(f"[4/4] Evaluando modos: {modes}")
    for mode in modes:
        fs_block = few_shot_block if mode == "few_shot" else ""
        results = evaluate(
            classifier, test_df,
            system_prompt=SYSTEM_PROMPT,
            few_shot_block=fs_block,
            mode_name=mode,
        )
        out_path = RESULTS_DIR / f"results_task1_gemini_{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Resultados guardados en {out_path}")

    print("\n✓ Evaluación completa.")


if __name__ == "__main__":
    main()
