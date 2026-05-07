# -*- coding: utf-8 -*-
"""
API REST para SciBETO-IMRaD — clasificación retórica IMRaD de fragmentos científicos en español.

Uso:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /clasificar   — clasifica un fragmento de texto
    GET  /health       — health check
"""

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_ID = "wiflore/SciBETO-IMRaD"
LABELS = ['INTRO', 'BACK', 'METH', 'RES', 'DISC', 'CONC', 'CONTR', 'LIM']

app = FastAPI(
    title="SciBETO-IMRaD API",
    description="Clasifica fragmentos de papers científicos en español en 8 categorías IMRaD.",
    version="1.0.0",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
if torch.cuda.is_available():
    model = model.cuda()


class TextIn(BaseModel):
    texto: str


class Prediction(BaseModel):
    etiqueta: str
    confianza: float
    probabilidades: dict


@app.post("/clasificar", response_model=Prediction)
def clasificar(body: TextIn):
    device = next(model.parameters()).device
    tokens = tokenizer(
        body.texto, truncation=True, max_length=512, return_tensors="pt"
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        probs = model(**tokens).logits.softmax(-1)[0].cpu()
    idx = probs.argmax().item()
    return Prediction(
        etiqueta=LABELS[idx],
        confianza=round(float(probs[idx]), 4),
        probabilidades={LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}
