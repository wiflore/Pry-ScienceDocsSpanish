"""Clasificación IMRaD con modelos de lenguaje (Qwen vía Ollama y Gemini)."""

import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import ollama
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

OLLAMA_MODEL = "qwen2.5:3b"
ETIQUETAS = ["Introducción", "Metodología", "Resultados", "Discusión"]

# ----- Esquema IMRaD-4 (clásico) -----
ALIAS_ETIQUETAS = {
    "introduccion": 0,
    "introducción": 0,
    "introduction": 0,
    "background": 0,
    "objective": 0,
    "metodologia": 1,
    "metodología": 1,
    "methods": 1,
    "method": 1,
    "resultados": 2,
    "results": 2,
    "discusion": 3,
    "discusión": 3,
    "discussion": 3,
    "conclusions": 3,
    "conclusiones": 3,
    "conclusión": 3,
    "conclusion": 3,
}

# ----- Esquema IMRaD-8 (anotación manual extendida) -----
# Orden de chequeo importante: prefijos largos primero para evitar
# que "conclusion" colisione con "conc" antes de tiempo.
ALIAS_ETIQUETAS_IMRAD8 = {
    "introduccion": 0, "introducción": 0, "introduction": 0, "intro": 0,
    "antecedentes": 1, "antecedente": 1, "background": 1, "marco teorico": 1,
    "marco teórico": 1, "back": 1,
    "metodologia": 2, "metodología": 2, "metodos": 2, "métodos": 2,
    "method": 2, "methods": 2, "meth": 2,
    "resultados": 3, "resultado": 3, "results": 3, "res": 3,
    "discusion": 4, "discusión": 4, "discussion": 4, "disc": 4,
    "conclusiones": 5, "conclusión": 5, "conclusion": 5, "conclusions": 5,
    "conc": 5,
    "contribuciones": 6, "contribución": 6, "contribuciones": 6,
    "contribuciones": 6, "aportes": 6, "contributions": 6, "contr": 6,
    "limitaciones": 7, "limitación": 7, "limitations": 7, "limitation": 7,
    "lim": 7,
}


def _parsear_etiqueta_generico(respuesta: str, alias: dict) -> Optional[int]:
    """Extrae una etiqueta usando el diccionario de alias dado."""
    texto = respuesta.strip().lower()
    limpia = texto.rstrip(".,;:!? \n\t")
    if limpia in alias:
        return alias[limpia]
    # Buscar la última mención de algún alias en el texto.
    # Ordenamos por longitud descendente para que "introducción" gane sobre "intro".
    mejor_pos = -1
    mejor_idx = None
    for a in sorted(alias.keys(), key=len, reverse=True):
        pos = texto.rfind(a)
        if pos > mejor_pos:
            mejor_pos = pos
            mejor_idx = alias[a]
    return mejor_idx


def _parsear_etiqueta(respuesta: str) -> Optional[int]:
    """Extrae la etiqueta IMRaD-4 (0–3) de la respuesta del modelo."""
    return _parsear_etiqueta_generico(respuesta, ALIAS_ETIQUETAS)


def _parsear_etiqueta_imrad8(respuesta: str) -> Optional[int]:
    """Extrae la etiqueta IMRaD-8 (0–7) de la respuesta del modelo."""
    return _parsear_etiqueta_generico(respuesta, ALIAS_ETIQUETAS_IMRAD8)


class QwenPrompter:
    """Clasifica secciones IMRaD usando Qwen a través de Ollama."""

    def __init__(self, model=OLLAMA_MODEL, prompt_mode="zero_shot",
                 prompts_dir="configs/prompts", temperature=0.0, n_votes=1,
                 label_scheme="imrad4"):
        self.model = model
        self.prompt_mode = prompt_mode
        self.temperature = temperature
        self.n_votes = n_votes
        self.label_scheme = label_scheme
        self._parser = _parsear_etiqueta_imrad8 if label_scheme == "imrad8" else _parsear_etiqueta
        self._fallback = 0

        archivo_prompt = Path(prompts_dir) / f"{prompt_mode}.txt"
        contenido = archivo_prompt.read_text(encoding="utf-8")

        # El prompt puede tener sistema y usuario separados por "---"
        if "\n---\n" in contenido:
            sistema, usuario = contenido.rsplit("\n---\n", 1)
            self.system_msg = sistema.strip()
            self.user_template = usuario.strip()
            self.use_chat = True
        else:
            self.system_msg = None
            self.user_template = contenido
            self.use_chat = False

    def _armar_prompt(self, texto: str) -> str:
        return self.user_template.replace("{text}", texto)

    def _consultar(self, texto: str, temperature: float) -> Optional[int]:
        """Hace una consulta al modelo y devuelve la etiqueta parseada."""
        user_content = self._armar_prompt(texto)

        if self.use_chat:
            es_qwen3 = "qwen3" in self.model.lower()
            if es_qwen3:
                # Modo sin razonamiento: más rápido para clasificación
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_msg},
                        {"role": "user", "content": user_content + " /no_think"},
                    ],
                    options={"temperature": max(temperature, 0.3), "top_p": 0.8,
                             "top_k": 20, "num_predict": 20},
                    think=False,
                )
            else:
                kwargs = dict(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_msg},
                        {"role": "user", "content": user_content},
                    ],
                    options={"temperature": temperature, "num_predict": 20},
                )
            respuesta = ollama.chat(**kwargs)
            raw = respuesta["message"]["content"]
        else:
            respuesta = ollama.generate(
                model=self.model,
                prompt=user_content,
                options={"temperature": temperature, "num_predict": 50},
            )
            raw = respuesta["response"]

        return self._parser(raw)

    def predict_one(self, texto: str) -> int:
        """Predice la etiqueta IMRaD para un texto. Retorna 0 si no puede parsear."""
        if self.n_votes <= 1:
            etiqueta = self._consultar(texto, self.temperature)
            return etiqueta if etiqueta is not None else self._fallback

        # Votación por mayoría con múltiples consultas
        votos = []
        temp = max(self.temperature, 0.3)
        for _ in range(self.n_votes):
            etiqueta = self._consultar(texto, temp)
            if etiqueta is not None:
                votos.append(etiqueta)
        return Counter(votos).most_common(1)[0][0] if votos else self._fallback

    def predict(self, textos: list) -> list:
        """Predice etiquetas para una lista de textos."""
        desc = f"Qwen ({self.prompt_mode})"
        if self.n_votes > 1:
            desc += f" x{self.n_votes}"

        predicciones = []
        latencias = []
        for texto in tqdm(textos, desc=desc):
            t0 = time.perf_counter()
            predicciones.append(self.predict_one(texto))
            latencias.append(time.perf_counter() - t0)

        arr = np.array(latencias)
        self.latency_stats = {
            "total_s": round(float(arr.sum()), 2),
            "mean_s": round(float(arr.mean()), 3),
            "median_s": round(float(np.median(arr)), 3),
            "std_s": round(float(arr.std()), 3),
            "samples": len(latencias),
        }
        return predicciones


class GeminiPrompter:
    """Clasifica secciones IMRaD usando la API de Gemini."""

    def __init__(self, model="gemini-2.0-flash", prompt_mode="zero_shot",
                 prompts_dir="configs/prompts", n_votes=1, api_key=None,
                 label_scheme="imrad4"):
        from google import genai

        self.model = model
        self.prompt_mode = prompt_mode
        self.n_votes = n_votes
        self.label_scheme = label_scheme
        self._parser = _parsear_etiqueta_imrad8 if label_scheme == "imrad8" else _parsear_etiqueta
        self._fallback = 0

        clave = api_key or os.environ.get("GEMINI_API_KEY")
        if not clave:
            raise ValueError("GEMINI_API_KEY no encontrada en entorno ni en .env")
        self._client = genai.Client(api_key=clave)

        archivo_prompt = Path(prompts_dir) / f"{prompt_mode}.txt"
        contenido = archivo_prompt.read_text(encoding="utf-8")

        if "\n---\n" in contenido:
            sistema, usuario = contenido.rsplit("\n---\n", 1)
            self.system_msg = sistema.strip()
            self.user_template = usuario.strip()
        else:
            self.system_msg = ""
            self.user_template = contenido

    def _armar_prompt(self, texto: str) -> str:
        return self.user_template.replace("{text}", texto)

    def _consultar(self, texto: str) -> Optional[int]:
        """Consulta la API de Gemini con reintentos en caso de rate limit."""
        from google.genai import types

        user_content = self._armar_prompt(texto)
        prompt_completo = f"{self.system_msg}\n\n{user_content}" if self.system_msg else user_content

        for intento in range(10):
            try:
                respuesta = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt_completo,
                    config=types.GenerateContentConfig(
                        temperature=0.0,
                        max_output_tokens=1024,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                # Extraer texto de la respuesta
                raw = ""
                if (respuesta.candidates and
                        respuesta.candidates[0].content and
                        respuesta.candidates[0].content.parts):
                    for parte in respuesta.candidates[0].content.parts:
                        if hasattr(parte, "text") and parte.text:
                            raw += parte.text
                if not raw:
                    raw = getattr(respuesta, "text", "") or ""
                return self._parser(raw)
            except Exception as e:
                err = str(e)
                if any(c in err for c in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"]):
                    espera = min(15 * (intento + 1), 90)
                    print(f"Rate limit (intento {intento+1}), esperando {espera}s...")
                    time.sleep(espera)
                else:
                    raise
        return None

    def predict_one(self, texto: str) -> int:
        """Predice la etiqueta IMRaD para un texto."""
        if self.n_votes <= 1:
            etiqueta = self._consultar(texto)
            return etiqueta if etiqueta is not None else self._fallback

        votos = [e for _ in range(self.n_votes) if (e := self._consultar(texto)) is not None]
        return Counter(votos).most_common(1)[0][0] if votos else self._fallback

    def predict(self, textos: list) -> list:
        """Predice etiquetas para una lista de textos."""
        desc = f"Gemini ({self.prompt_mode})"
        if self.n_votes > 1:
            desc += f" x{self.n_votes}"

        predicciones = []
        latencias = []
        for texto in tqdm(textos, desc=desc):
            t0 = time.perf_counter()
            predicciones.append(self.predict_one(texto))
            latencias.append(time.perf_counter() - t0)

        arr = np.array(latencias)
        self.latency_stats = {
            "total_s": round(float(arr.sum()), 2),
            "mean_s": round(float(arr.mean()), 3),
            "median_s": round(float(np.median(arr)), 3),
            "std_s": round(float(arr.std()), 3),
            "samples": len(latencias),
        }
        return predicciones
