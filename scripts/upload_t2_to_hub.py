# -*- coding: utf-8 -*-
"""
Sube el modelo SciBETO T2 (binario contribucion) a HuggingFace Hub.

Uso:
    python scripts/upload_t2_to_hub.py

Requisitos:
    - Token HF con permisos de escritura en .env  →  HF_TOKEN=hf_xxxx
    - O haber ejecutado previamente: huggingface-cli login
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv(override=True)

# ── Configuración ──────────────────────────────────────────────────────────────
LOCAL_PATH  = Path("models/scibeto-task2-binario/best_model")
HF_REPO_ID  = "wiflore/SciBETO-T2-contribucion"
COMMIT_MSG  = "Upload SciBETO-large fine-tuned for binary contribution detection (T2)"
HF_TOKEN    = os.getenv("HuggingFace", "")

if not HF_TOKEN:
    raise EnvironmentError("No se encontró la variable 'HuggingFace' en .env. Agrega HuggingFace=hf_xxxx")

# ── Carga local ────────────────────────────────────────────────────────────────
print("Cargando tokenizer y modelo desde disco...")
tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_PATH))
model     = AutoModelForSequenceClassification.from_pretrained(str(LOCAL_PATH))

# ── Upload ─────────────────────────────────────────────────────────────────────
kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

print(f"\nSubiendo tokenizer a {HF_REPO_ID}...")
tokenizer.push_to_hub(HF_REPO_ID, commit_message=COMMIT_MSG, **kwargs)

print(f"Subiendo modelo (puede tardar varios minutos con el .safetensors)...")
model.push_to_hub(HF_REPO_ID, commit_message=COMMIT_MSG, **kwargs)

print(f"\n✓ Listo. Modelo disponible en:")
print(f"  https://huggingface.co/{HF_REPO_ID}")
print(f"\nPara cargarlo desde la API cambiar T2_MODEL_PATH por:")
print(f'  T2_MODEL_PATH = "{HF_REPO_ID}"')
