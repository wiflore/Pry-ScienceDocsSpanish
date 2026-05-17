# -*- coding: utf-8 -*-
"""
API REST — SciBETO IMRaD + Tarea 2 contribución científica.

Uso:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /clasificar    — T1: clasifica fragmento en 8 clases IMRaD
                          modelo: "scibeto" (default)
    POST /contribucion  — T2: clasifica fragmento como contribucion / no_contribucion
                          modelo: "scibeto" (default) | "gemini"
    GET  /health        — health check
    GET  /modelos       — lista modelos disponibles por tarea
"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

# ── Constantes ────────────────────────────────────────────────────────────────
T1_MODEL_ID   = "wiflore/SciBETO-IMRaD"
T1_LABELS     = ['INTRO', 'BACK', 'METH', 'RES', 'DISC', 'CONC', 'CONTR', 'LIM']

T2_MODEL_PATH = "models/scibeto-task2-binario/best_model"
T2_LABELS     = ['no_contribucion', 'contribucion']
T2_HEAD, T2_TAIL = 128, 382

GEMINI_MODEL  = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

T2_SYSTEM_PROMPT = """Eres un clasificador experto de fragmentos de artículos científicos en español.

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

Marcar como NO CONTRIBUCIÓN cuando:
- describe contexto, antecedentes o resultados sin declarar aporte propio
- menciona contribuciones de OTROS autores
- declara objetivos sin novedad explícita
- resume trabajos previos"""

_ALIAS_T2 = {
    "contribucion": 1, "contribución": 1, "contribuciones": 1,
    "contribution": 1, "si": 1, "yes": 1, "1": 1,
    "no_contribucion": 0, "no contribucion": 0, "no_contribución": 0,
    "no contribución": 0, "no": 0, "0": 0,
}

AVAILABLE_MODELS = {
    "t1": ["scibeto"],
    "t2": ["scibeto", "gemini"],
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SciBETO-IMRaD + Contribución API",
    description=(
        "T1 (/clasificar): clasifica fragmentos en 8 categorías IMRaD. "
        "T2 (/contribucion): detecta si un fragmento es una contribución científica."
    ),
    version="2.0.0",
)

# ── Carga de modelos encoder ──────────────────────────────────────────────────
t1_tokenizer = AutoTokenizer.from_pretrained(T1_MODEL_ID)
t1_model = AutoModelForSequenceClassification.from_pretrained(T1_MODEL_ID)
t1_model.eval()

t2_tokenizer = AutoTokenizer.from_pretrained(T2_MODEL_PATH)
t2_model = AutoModelForSequenceClassification.from_pretrained(T2_MODEL_PATH)
t2_model.eval()

if torch.cuda.is_available():
    t1_model = t1_model.cuda()
    t2_model = t2_model.cuda()

# ── Schemas ───────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    texto: str
    modelo: str = "scibeto"


class Prediction(BaseModel):
    etiqueta: str
    confianza: float
    probabilidades: dict
    modelo_usado: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def _tokenize_head_tail(tokenizer, texto: str, head: int, tail: int, device):
    """Head+Tail truncation: primeros `head` + últimos `tail` tokens."""
    ids = tokenizer.encode(texto, add_special_tokens=False)
    max_tokens = head + tail
    if len(ids) > max_tokens:
        ids = ids[:head] + ids[-tail:]
    ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
    input_ids = torch.tensor([ids], dtype=torch.long).to(device)
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def _predict_t2_gemini(texto: str) -> Prediction:
    """Llama a Gemini 2.5 Flash para clasificación T2."""
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY no configurada en el entorno."
        )
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="SDK google-genai no instalado. Ejecuta: pip install google-genai"
        )

    client = genai.Client(api_key=GEMINI_API_KEY)
    user_content = f'{T2_SYSTEM_PROMPT}\n\nFragmento: "{texto}"\nEtiqueta:'

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=15,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = response.text.strip().lower()
    label_idx = _ALIAS_T2.get(raw, -1)

    if label_idx == -1:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini devolvió respuesta inesperada: '{raw}'"
        )

    label = T2_LABELS[label_idx]
    # Gemini no da probabilidades — devolvemos confianza nominal
    conf = 0.95 if label_idx == 1 else 0.95
    probs = {T2_LABELS[1 - label_idx]: round(1 - conf, 4), label: round(conf, 4)}
    return Prediction(
        etiqueta=label,
        confianza=conf,
        probabilidades=probs,
        modelo_usado=GEMINI_MODEL,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/clasificar", response_model=Prediction)
def clasificar(body: TextIn):
    """T1: clasificación retórica IMRaD (8 clases). Modelo: scibeto."""
    if body.modelo != "scibeto":
        raise HTTPException(
            status_code=422,
            detail=f"Modelo '{body.modelo}' no disponible para T1. "
                   f"Disponibles: {AVAILABLE_MODELS['t1']}"
        )
    device = next(t1_model.parameters()).device
    tokens = t1_tokenizer(
        body.texto, truncation=True, max_length=512, return_tensors="pt"
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        probs = t1_model(**tokens).logits.softmax(-1)[0].cpu()
    idx = probs.argmax().item()
    return Prediction(
        etiqueta=T1_LABELS[idx],
        confianza=round(float(probs[idx]), 4),
        probabilidades={T1_LABELS[i]: round(float(probs[i]), 4) for i in range(len(T1_LABELS))},
        modelo_usado=T1_MODEL_ID,
    )


@app.post("/contribucion", response_model=Prediction)
def contribucion(body: TextIn):
    """T2: detección binaria de contribución científica. Modelo: scibeto | gemini."""
    if body.modelo not in AVAILABLE_MODELS["t2"]:
        raise HTTPException(
            status_code=422,
            detail=f"Modelo '{body.modelo}' no disponible para T2. "
                   f"Disponibles: {AVAILABLE_MODELS['t2']}"
        )

    if body.modelo == "gemini":
        return _predict_t2_gemini(body.texto)

    # scibeto (default)
    device = next(t2_model.parameters()).device
    tokens = _tokenize_head_tail(t2_tokenizer, body.texto, T2_HEAD, T2_TAIL, device)
    with torch.no_grad():
        probs = t2_model(**tokens).logits.softmax(-1)[0].cpu()
    idx = probs.argmax().item()
    return Prediction(
        etiqueta=T2_LABELS[idx],
        confianza=round(float(probs[idx]), 4),
        probabilidades={T2_LABELS[i]: round(float(probs[i]), 4) for i in range(len(T2_LABELS))},
        modelo_usado=T2_MODEL_PATH,
    )


@app.get("/modelos")
def modelos():
    """Lista los modelos disponibles por tarea."""
    return {
        "t1": {"endpoint": "/clasificar",   "modelos": AVAILABLE_MODELS["t1"]},
        "t2": {"endpoint": "/contribucion", "modelos": AVAILABLE_MODELS["t2"]},
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "t1_model": T1_MODEL_ID,
        "t2_model": T2_MODEL_PATH,
        "gemini_key_set": bool(GEMINI_API_KEY),
    }
