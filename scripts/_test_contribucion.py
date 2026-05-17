import requests

BASE = "http://127.0.0.1:8000"
T1 = "Este trabajo propone un nuevo corpus anotado de articulos cientificos en espanol para la clasificacion retorica automatica."
NEG = "Varios estudios previos han utilizado BERT para clasificar textos cientificos en ingles con resultados satisfactorios."

cases = [
    ("contribucion gemini positivo",  "/contribucion", {"texto": T1,  "modelo": "gemini"}),
    ("contribucion scibeto negativo", "/contribucion", {"texto": NEG, "modelo": "scibeto"}),
    ("contribucion gemini negativo",  "/contribucion", {"texto": NEG, "modelo": "gemini"}),
    ("contribucion modelo invalido",  "/contribucion", {"texto": T1,  "modelo": "gpt4"}),
]

for name, path, body in cases:
    r = requests.post(f"{BASE}{path}", json=body)
    d = r.json()
    if r.status_code == 200:
        print(f"[{r.status_code}] {name}: etiqueta={d['etiqueta']} conf={d['confianza']} modelo={d['modelo_usado']}")
    else:
        print(f"[{r.status_code}] {name}: {d.get('detail')}")
