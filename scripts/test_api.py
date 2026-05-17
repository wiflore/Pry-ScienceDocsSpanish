# -*- coding: utf-8 -*-
"""Script de prueba para todos los endpoints de la API."""
import requests
import json

BASE = "http://127.0.0.1:8000"

TEXTO_T1 = "Este trabajo propone un nuevo corpus anotado de articulos cientificos en espanol para la clasificacion retorica automatica."

TEXTO_MULTI = (
    "Este trabajo presenta un nuevo metodo para la clasificacion retorica automatica de articulos cientificos en espanol mediante modelos de lenguaje pre-entrenados.\n\n"
    "La clasificacion retorica ha sido estudiada ampliamente en ingles, pero existen pocos recursos para el espanol. Trabajos previos como SciBERT han demostrado que los modelos pre-entrenados son efectivos para esta tarea.\n\n"
    "Se utilizo un corpus de 500 articulos anotados manualmente con 8 categorias IMRaD. El modelo fue entrenado durante 3 epocas con un learning rate de 2e-5 y batch size de 16.\n\n"
    "Los resultados muestran que el modelo alcanza un F1-macro de 0.84, superando a los baselines de TF-IDF y los modelos de prompting en cero disparos.\n\n"
    "La principal contribucion de este trabajo es un corpus anotado de fragmentos cientificos en espanol y un modelo fine-tuneado disponible publicamente para la clasificacion retorica."
)


def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


sep("GET /health")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("GET /modelos")
r = requests.get(f"{BASE}/modelos")
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("POST /clasificar - scibeto")
r = requests.post(f"{BASE}/clasificar", json={"texto": TEXTO_T1, "modelo": "scibeto"})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("POST /clasificar - gemini")
r = requests.post(f"{BASE}/clasificar", json={"texto": TEXTO_T1, "modelo": "gemini"})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("POST /clasificar - modelo invalido")
r = requests.post(f"{BASE}/clasificar", json={"texto": TEXTO_T1, "modelo": "qwen"})
print(r.status_code, r.json().get("detail"))

sep("POST /contribucion - scibeto (positivo)")
r = requests.post(f"{BASE}/contribucion", json={"texto": TEXTO_T1, "modelo": "scibeto"})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("POST /analizar - scibeto (5 parrafos)")
r = requests.post(f"{BASE}/analizar", json={"texto": TEXTO_MULTI, "modelo": "scibeto"})
data = r.json()
print(f"n_fragmentos: {data['n_fragmentos']}  modelo: {data['modelo']}")
for frag in data["fragmentos"]:
    marker = "[CONTRIB]" if frag["t2"] == "contribucion" else "         "
    print(f"  {marker} T1={frag['t1']:5s}({frag['confianza_t1']:.2f}) | T2={frag['t2']}({frag['confianza_t2']:.2f}) | {frag['fragmento'][:55]}...")

sep("POST /analizar - modelo invalido")
r = requests.post(f"{BASE}/analizar", json={"texto": TEXTO_MULTI, "modelo": "qwen"})
print(r.status_code, r.json().get("detail"))

# ── Casos faltantes ───────────────────────────────────────────────────────────

TEXTO_NEG = "Varios estudios previos han utilizado BERT para clasificar textos cientificos en ingles con resultados satisfactorios."

sep("POST /contribucion - gemini (positivo)")
r = requests.post(f"{BASE}/contribucion", json={"texto": TEXTO_T1, "modelo": "gemini"})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("POST /contribucion - scibeto (negativo)")
r = requests.post(f"{BASE}/contribucion", json={"texto": TEXTO_NEG, "modelo": "scibeto"})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("POST /contribucion - gemini (negativo)")
r = requests.post(f"{BASE}/contribucion", json={"texto": TEXTO_NEG, "modelo": "gemini"})
print(json.dumps(r.json(), indent=2, ensure_ascii=False))

sep("POST /contribucion - modelo invalido")
r = requests.post(f"{BASE}/contribucion", json={"texto": TEXTO_T1, "modelo": "gpt4"})
print(r.status_code, r.json().get("detail"))

sep("POST /analizar - gemini (5 parrafos)")
r = requests.post(f"{BASE}/analizar", json={"texto": TEXTO_MULTI, "modelo": "gemini"})
data = r.json()
print(f"n_fragmentos: {data['n_fragmentos']}  modelo: {data['modelo']}")
for frag in data["fragmentos"]:
    marker = "[CONTRIB]" if frag["t2"] == "contribucion" else "         "
    print(f"  {marker} T1={frag['t1']:5s}({frag['confianza_t1']:.2f}) | T2={frag['t2']}({frag['confianza_t2']:.2f}) | {frag['fragmento'][:55]}...")

sep("POST /analizar - texto corto (borde: 1 parrafo valido)")
r = requests.post(f"{BASE}/analizar", json={"texto": "Este estudio propone un metodo nuevo.", "modelo": "scibeto"})
data = r.json()
print(f"n_fragmentos: {data['n_fragmentos']}")
print(json.dumps(data["fragmentos"][0], indent=2, ensure_ascii=False))

sep("POST /analizar - texto vacio (debe dar 422)")
r = requests.post(f"{BASE}/analizar", json={"texto": "   ", "modelo": "scibeto"})
print(r.status_code, r.json().get("detail"))
