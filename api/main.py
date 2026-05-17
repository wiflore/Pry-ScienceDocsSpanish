# -*- coding: utf-8 -*-
"""
API REST — SciBETO IMRaD + Tarea 2 contribución científica.

Uso:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /clasificar    — T1: clasifica fragmento en 8 clases IMRaD
    POST /contribucion  — T2: clasifica fragmento como contribucion / no_contribucion
    GET  /health        — health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ── Tarea 1 ───────────────────────────────────────────────────────────────────
T1_MODEL_ID = "wiflore/SciBETO-IMRaD"
T1_LABELS   = ['INTRO', 'BACK', 'METH', 'RES', 'DISC', 'CONC', 'CONTR', 'LIM']

# ── Tarea 2 ───────────────────────────────────────────────────────────────────
T2_MODEL_PATH = "models/scibeto-task2-binario/best_model"
T2_LABELS     = ['no_contribucion', 'contribucion']
# Head+Tail: primeros 128 + últimos 382 tokens (igual que en entrenamiento)
T2_HEAD = 128
T2_TAIL = 382

app = FastAPI(
    title="SciBETO-IMRaD + Contribución API",
    description=(
        "T1: clasifica fragmentos científicos en 8 categorías IMRaD. "
        "T2: detecta si un fragmento es una contribución científica."
    ),
    version="2.0.0",
)

# ── Carga de modelos ──────────────────────────────────────────────────────────
t1_tokenizer = AutoTokenizer.from_pretrained(T1_MODEL_ID)
t1_model     = AutoModelForSequenceClassification.from_pretrained(T1_MODEL_ID)
t1_model.eval()

t2_tokenizer = AutoTokenizer.from_pretrained(T2_MODEL_PATH)
t2_model     = AutoModelForSequenceClassification.from_pretrained(T2_MODEL_PATH)
t2_model.eval()

if torch.cuda.is_available():
    t1_model = t1_model.cuda()
    t2_model = t2_model.cuda()


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextIn(BaseModel):
    texto: str


class Prediction(BaseModel):
    etiqueta: str
    confianza: float
    probabilidades: dict


# ── Helper Head+Tail para T2 ──────────────────────────────────────────────────
def _tokenize_head_tail(tokenizer, texto: str, head: int, tail: int, device):
    """Tokeniza con estrategia Head+Tail: primeros `head` + últimos `tail` tokens."""
    ids = tokenizer.encode(texto, add_special_tokens=False)
    max_tokens = head + tail
    if len(ids) > max_tokens:
        ids = ids[:head] + ids[-tail:]
    # Añadir tokens especiales [CLS] ... [SEP]
    ids = [tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id]
    input_ids      = torch.tensor([ids], dtype=torch.long).to(device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/clasificar", response_model=Prediction)
def clasificar(body: TextIn):
    """T1: clasificación retórica IMRaD (8 clases)."""
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
    )


@app.post("/contribucion", response_model=Prediction)
def contribucion(body: TextIn):
    """T2: detección binaria de contribución científica."""
    device = next(t2_model.parameters()).device
    tokens = _tokenize_head_tail(t2_tokenizer, body.texto, T2_HEAD, T2_TAIL, device)
    with torch.no_grad():
        probs = t2_model(**tokens).logits.softmax(-1)[0].cpu()
    idx = probs.argmax().item()
    return Prediction(
        etiqueta=T2_LABELS[idx],
        confianza=round(float(probs[idx]), 4),
        probabilidades={T2_LABELS[i]: round(float(probs[i]), 4) for i in range(len(T2_LABELS))},
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "t1_model": T1_MODEL_ID,
        "t2_model": T2_MODEL_PATH,
    }
