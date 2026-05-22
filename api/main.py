# -*- coding: utf-8 -*-
"""
API REST — SciBETO IMRaD + Tarea 2 contribución científica.

Uso:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    POST /clasificar    — T1: clasifica fragmento en 8 clases IMRaD
                          modelo: "scibeto" (default) | "gemini" | "qwen"
    POST /contribucion  — T2: clasifica fragmento como contribucion / no_contribucion
                          modelo: "scibeto" (default) | "gemini" | "qwen"
    POST /analizar      — Pipeline: segmenta texto completo y aplica T1+T2 por párrafo
                          modelo: "scibeto" (default) | "gemini" | "qwen"
    GET  /health        — health check
    GET  /modelos       — lista modelos disponibles por tarea

Requisito para modelo "qwen": OPENROUTER_API_KEY configurada en .env.
    Modelo por defecto: qwen/qwen3-8b:free (configurable con OPENROUTER_MODEL).
"""

import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

# ── Constantes ────────────────────────────────────────────────────────────────
T1_MODEL_ID   = "wiflore/SciBETO-IMRaD"
T1_LABELS     = ['INTRO', 'BACK', 'METH', 'RES', 'DISC', 'CONC', 'CONTR', 'LIM']

T2_MODEL_PATH = "wiflore/SciBETO-T2-contribucion"
T2_LABELS     = ['no_contribucion', 'contribucion']
T2_HEAD, T2_TAIL = 128, 382

GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")

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

T1_SYSTEM_PROMPT = """Eres un clasificador experto de fragmentos de artículos científicos en español.

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

T1_ALIAS_MAP = {
    "back": "BACK", "conc": "CONC", "contr": "CONTR", "disc": "DISC",
    "intro": "INTRO", "lim": "LIM", "meth": "METH", "res": "RES",
    "antecedentes": "BACK", "antecedente": "BACK", "estado del arte": "BACK",
    "conclusiones": "CONC", "conclusión": "CONC", "conclusion": "CONC",
    "contribuciones": "CONTR", "contribución": "CONTR", "contribution": "CONTR",
    "discusión": "DISC", "discusion": "DISC", "discussion": "DISC",
    "introducción": "INTRO", "introduccion": "INTRO", "introduction": "INTRO",
    "limitaciones": "LIM", "limitación": "LIM", "limitation": "LIM",
    "metodología": "METH", "metodologia": "METH", "methodology": "METH",
    "resultados": "RES", "results": "RES", "resultado": "RES",
    "background": "BACK", "limitations": "LIM", "methods": "METH",
    "n/a": "OTRO", "na": "OTRO", "otro": "OTRO", "other": "OTRO",
    "ninguno": "OTRO", "none": "OTRO", "no aplica": "OTRO",
}

AVAILABLE_MODELS = {
    "t1":       ["scibeto", "gemini", "qwen"],
    "t2":       ["scibeto", "gemini", "qwen"],
    "pipeline": ["scibeto", "gemini", "qwen"],
}

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SciBETO-IMRaD + Contribución API",
    description=(
        "T1 (/clasificar): clasifica fragmentos en 8 categorías IMRaD. "
        "T2 (/contribucion): detecta si un fragmento es una contribución científica."
    ),
    version="2.1.0",
)

CORS_ORIGINS = [o.strip() for o in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://app.prysciencedocs.xyz,https://prysciencedocs.xyz"
).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https://.*\.amplifyapp\.com$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
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


class AnalisisIn(BaseModel):
    texto: str
    modelo: str = "scibeto"


class FragmentoResult(BaseModel):
    fragmento: str
    t1: str
    confianza_t1: float
    probabilidades_t1: dict
    t2: str
    confianza_t2: float
    probabilidades_t2: dict
    modelo_t1: str
    modelo_t2: str


class AnalisisOut(BaseModel):
    n_fragmentos: int
    modelo: str
    fragmentos: list[FragmentoResult]


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


# ── Helpers encoder ──────────────────────────────────────────────────────────
def _predict_t1_scibeto(texto: str) -> Prediction:
    """SciBETO-IMRaD para T1 (8 clases IMRaD)."""
    device = next(t1_model.parameters()).device
    tokens = t1_tokenizer(texto, truncation=True, max_length=512, return_tensors="pt")
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


def _predict_t1_gemini(texto: str) -> Prediction:
    """Gemini 2.5 Flash para T1 (8 clases IMRaD)."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY no configurada.")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise HTTPException(status_code=503, detail="SDK google-genai no instalado.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    full_prompt = f'{T1_SYSTEM_PROMPT}\n\nFragmento: "{texto}"\nEtiqueta:'
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=10,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    raw = (response.text or "").strip().lower()
    for ch in ".,;:!?\n\t \"'":
        raw = raw.rstrip(ch)
    label = T1_ALIAS_MAP.get(raw)
    if label is None:
        for alias, lbl in sorted(T1_ALIAS_MAP.items(), key=lambda x: -len(x[0])):
            if alias in raw:
                label = lbl
                break
    if label is None:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini devolvió respuesta inesperada para T1: '{raw}'"
        )
    if label == "OTRO":
        return Prediction(etiqueta="OTRO", confianza=1.0, probabilidades={"OTRO": 1.0}, modelo_usado=GEMINI_MODEL)
    conf = 0.95
    probs = {lbl: round(0.05 / (len(T1_LABELS) - 1), 4) for lbl in T1_LABELS}
    probs[label] = conf
    return Prediction(etiqueta=label, confianza=conf, probabilidades=probs, modelo_usado=GEMINI_MODEL)


def _predict_t2_scibeto(texto: str) -> Prediction:
    """SciBETO fine-tuneado para T2 (binario: contribucion / no_contribucion)."""
    device = next(t2_model.parameters()).device
    tokens = _tokenize_head_tail(t2_tokenizer, texto, T2_HEAD, T2_TAIL, device)
    with torch.no_grad():
        probs = t2_model(**tokens).logits.softmax(-1)[0].cpu()
    idx = probs.argmax().item()
    return Prediction(
        etiqueta=T2_LABELS[idx],
        confianza=round(float(probs[idx]), 4),
        probabilidades={T2_LABELS[i]: round(float(probs[i]), 4) for i in range(len(T2_LABELS))},
        modelo_usado=T2_MODEL_PATH,
    )


# ── Helpers Qwen (OpenRouter) ────────────────────────────────────────────────
def _openrouter_chat(prompt: str, max_tokens: int = 15) -> str:
    """Envía un prompt a OPENROUTER_MODEL via OpenRouter y devuelve el texto limpio.
    Añade /no_think para desactivar el chain-of-thought de Qwen3.
    Elimina bloques <think>...</think> si el modelo los incluye."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY no configurada en el entorno."
        )
    try:
        from openai import OpenAI
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="SDK openai no instalado. Ejecuta: pip install openai"
        )
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        resp = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt + " /no_think"}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"OpenRouter no disponible: {exc}"
        )
    # Eliminar bloque <think>…</think> por si Qwen3 lo incluye de todas formas
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


def _predict_t1_qwen(texto: str) -> Prediction:
    """Qwen3-8B via OpenRouter para T1 (8 clases IMRaD)."""
    full_prompt = f'{T1_SYSTEM_PROMPT}\n\nFragmento: "{texto}"\nEtiqueta:'
    raw = _openrouter_chat(full_prompt, max_tokens=15).lower()
    for ch in ".,;:!?\n\t \"'":
        raw = raw.rstrip(ch)
    label = T1_ALIAS_MAP.get(raw)
    if label is None:
        for alias, lbl in sorted(T1_ALIAS_MAP.items(), key=lambda x: -len(x[0])):
            if alias in raw:
                label = lbl
                break
    if label is None:
        raise HTTPException(
            status_code=502,
            detail=f"Qwen devolvió respuesta inesperada para T1: '{raw}'"
        )
    if label == "OTRO":
        return Prediction(etiqueta="OTRO", confianza=1.0, probabilidades={"OTRO": 1.0}, modelo_usado=OPENROUTER_MODEL)
    conf = 0.90
    probs = {lbl: round(0.10 / (len(T1_LABELS) - 1), 4) for lbl in T1_LABELS}
    probs[label] = conf
    return Prediction(etiqueta=label, confianza=conf, probabilidades=probs, modelo_usado=OPENROUTER_MODEL)


def _predict_t2_qwen(texto: str) -> Prediction:
    """Qwen3-8B via OpenRouter para T2 (binario: contribucion / no_contribucion)."""
    full_prompt = f'{T2_SYSTEM_PROMPT}\n\nFragmento: "{texto}"\nEtiqueta:'
    raw = _openrouter_chat(full_prompt, max_tokens=15).lower()
    label_idx = _ALIAS_T2.get(raw, -1)
    if label_idx == -1:
        raise HTTPException(
            status_code=502,
            detail=f"Qwen devolvió respuesta inesperada para T2: '{raw}'"
        )
    label = T2_LABELS[label_idx]
    conf = 0.90
    probs = {T2_LABELS[1 - label_idx]: round(1 - conf, 4), label: round(conf, 4)}
    return Prediction(etiqueta=label, confianza=conf, probabilidades=probs, modelo_usado=OPENROUTER_MODEL)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/clasificar", response_model=Prediction)
def clasificar(body: TextIn):
    """T1: clasificación retórica IMRaD (8 clases). Modelo: scibeto | gemini | qwen."""
    if body.modelo not in AVAILABLE_MODELS["t1"]:
        raise HTTPException(
            status_code=422,
            detail=f"Modelo '{body.modelo}' no disponible para T1. "
                   f"Disponibles: {AVAILABLE_MODELS['t1']}"
        )
    if body.modelo == "gemini":
        return _predict_t1_gemini(body.texto)
    if body.modelo == "qwen":
        return _predict_t1_qwen(body.texto)
    return _predict_t1_scibeto(body.texto)


@app.post("/contribucion", response_model=Prediction)
def contribucion(body: TextIn):
    """T2: detección binaria de contribución científica. Modelo: scibeto | gemini | qwen."""
    if body.modelo not in AVAILABLE_MODELS["t2"]:
        raise HTTPException(
            status_code=422,
            detail=f"Modelo '{body.modelo}' no disponible para T2. "
                   f"Disponibles: {AVAILABLE_MODELS['t2']}"
        )
    if body.modelo == "gemini":
        return _predict_t2_gemini(body.texto)
    if body.modelo == "qwen":
        return _predict_t2_qwen(body.texto)
    return _predict_t2_scibeto(body.texto)


def _segmentar(texto: str, min_palabras: int = 250, max_palabras: int = 1000) -> list[str]:
    """Segmenta texto en chunks de [min_palabras, max_palabras] palabras.
    1. Normaliza saltos de línea (\r\n, \r, múltiples \n).
    2. Divide por párrafos: doble salto si existe, sino salto simple.
    3. Fusiona párrafos consecutivos hasta alcanzar min_palabras.
    4. Si un bloque supera max_palabras, lo subdivide en chunks de ese tamaño."""
    # Normalizar saltos de línea: \r\n y \r → \n, luego colapsar 3+ \n en \n\n
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    texto = re.sub(r'\n{3,}', '\n\n', texto)
    # Elegir separador: doble salto si existe, sino salto simple
    separador = "\n\n" if "\n\n" in texto else "\n"
    partes = [p.strip() for p in texto.split(separador) if p.strip()]
    # Normalizar líneas internas (por si quedan \n simples dentro de un párrafo)
    partes = [" ".join(ln.strip() for ln in p.split("\n") if ln.strip()) for p in partes]

    # Fusionar párrafos cortos con el siguiente hasta alcanzar min_palabras
    merged: list[str] = []
    buffer = ""
    for parte in partes:
        if buffer:
            buffer = buffer + " " + parte
        else:
            buffer = parte
        if len(buffer.split()) >= min_palabras:
            merged.append(buffer)
            buffer = ""
    if buffer and len(buffer.split()) >= min_palabras:
        merged.append(buffer)
    elif buffer and merged:
        # residuo menor a min_palabras: anexar al último bloque
        merged[-1] = merged[-1] + " " + buffer
    elif buffer:
        # texto completo < min_palabras: enviar igualmente si tiene algo
        if len(buffer.split()) >= 5:
            merged.append(buffer)

    # Subdividir bloques que superen max_palabras
    fragmentos: list[str] = []
    for bloque in merged:
        palabras = bloque.split()
        if len(palabras) <= max_palabras:
            fragmentos.append(bloque)
        else:
            for i in range(0, len(palabras), max_palabras):
                chunk = " ".join(palabras[i:i + max_palabras])
                if len(chunk.split()) >= min_palabras:
                    fragmentos.append(chunk)
                elif fragmentos:
                    fragmentos[-1] = fragmentos[-1] + " " + chunk
    return fragmentos


@app.post("/analizar", response_model=AnalisisOut)
def analizar(body: AnalisisIn):
    """Pipeline completo: segmenta texto y aplica T1 + T2 a cada párrafo.
    Modelo: scibeto | gemini | qwen."""
    if body.modelo not in AVAILABLE_MODELS["pipeline"]:
        raise HTTPException(
            status_code=422,
            detail=f"Modelo '{body.modelo}' no disponible para pipeline. "
                   f"Disponibles: {AVAILABLE_MODELS['pipeline']}"
        )
    fragmentos_texto = _segmentar(body.texto)
    if not fragmentos_texto:
        raise HTTPException(
            status_code=422,
            detail="El texto no contiene párrafos válidos (mínimo 5 palabras cada uno)."
        )
    _pred_t1 = (_predict_t1_gemini if body.modelo == "gemini"
                else _predict_t1_qwen if body.modelo == "qwen"
                else _predict_t1_scibeto)
    _pred_t2 = (_predict_t2_gemini if body.modelo == "gemini"
                else _predict_t2_qwen if body.modelo == "qwen"
                else _predict_t2_scibeto)
    resultados: list[FragmentoResult] = []
    for frag in fragmentos_texto:
        p1 = _pred_t1(frag)
        p2 = _pred_t2(frag)
        resultados.append(FragmentoResult(
            fragmento=frag,
            t1=p1.etiqueta,
            confianza_t1=p1.confianza,
            probabilidades_t1=p1.probabilidades,
            t2=p2.etiqueta,
            confianza_t2=p2.confianza,
            probabilidades_t2=p2.probabilidades,
            modelo_t1=p1.modelo_usado,
            modelo_t2=p2.modelo_usado,
        ))
    return AnalisisOut(n_fragmentos=len(resultados), modelo=body.modelo, fragmentos=resultados)


@app.get("/modelos")
def modelos():
    """Lista los modelos disponibles por tarea y endpoint."""
    return {
        "t1":       {"endpoint": "/clasificar",   "modelos": AVAILABLE_MODELS["t1"]},
        "t2":       {"endpoint": "/contribucion", "modelos": AVAILABLE_MODELS["t2"]},
        "pipeline": {"endpoint": "/analizar",     "modelos": AVAILABLE_MODELS["pipeline"]},
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "t1_model": T1_MODEL_ID,
        "t2_model": T2_MODEL_PATH,
        "gemini_key_set": bool(GEMINI_API_KEY),
    }
