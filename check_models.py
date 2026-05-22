"""Consulta modelos disponibles en OpenRouter y sus límites/precios."""
import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))

# Modelos candidatos para clasificación de texto corto
candidates = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-8b",
    "qwen/qwen3-8b:free",
    "qwen/qwen3-14b",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-12b-it",
]

print("Probando modelos con prompt corto...\n")
prompt = 'Responde SOLO con la etiqueta: INTRO, BACK, METH, RES, DISC, CONC, CONTR o LIM.\n\nFragmento: "En este trabajo proponemos un método de clasificación retórica."\nEtiqueta: /no_think'

for model in candidates:
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        ans = r.choices[0].message.content.strip()
        usage = r.usage
        print(f"✅ {model}")
        print(f"   Respuesta: {ans}")
        if usage:
            print(f"   Tokens: {usage.prompt_tokens} in + {usage.completion_tokens} out")
        print()
    except Exception as e:
        print(f"❌ {model}: {e}\n")
