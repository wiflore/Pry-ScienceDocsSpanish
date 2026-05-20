"""
Prueba del endpoint /analizar con modelo=qwen usando el docx de prueba.
Uso: python scripts/_test_qwen_docx.py
"""
import json
import sys
import requests
import docx  # pip install python-docx

DOCX_PATH = r"C:\Users\adrso\Downloads\artEspañolPrueba.docx"
API_URL = "http://127.0.0.1:8000/analizar"


def extract_text(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def main():
    print(f"Leyendo {DOCX_PATH}...")
    texto = extract_text(DOCX_PATH)
    palabras = len(texto.split())
    print(f"Texto extraído: {palabras} palabras\n")
    print("─" * 60)

    payload = {"texto": texto, "modelo": "qwen"}
    print("Enviando a /analizar con modelo=qwen...")
    print("(Primera petición carga Qwen3-8B — puede tardar ~60s)\n")

    try:
        resp = requests.post(API_URL, json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("ERROR: No se puede conectar a la API. ¿Está uvicorn corriendo?")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"ERROR HTTP {resp.status_code}: {resp.text}")
        sys.exit(1)

    data = resp.json()
    print(f"Fragmentos procesados: {data['n_fragmentos']}")
    print(f"Modelo usado: {data['modelo']}\n")

    for i, frag in enumerate(data["fragmentos"], 1):
        t2_flag = "✓ CONTRIBUCION" if frag["t2"] == "contribucion" else "  no_contribucion"
        print(f"[{i:02d}] T1={frag['t1']:5s} ({frag['confianza_t1']:.2f}) | T2={t2_flag} ({frag['confianza_t2']:.2f})")
        snippet = " ".join(frag["fragmento"].split()[:15])
        print(f"     {snippet}...")
        print()


if __name__ == "__main__":
    main()
