"""
Pipeline de Extracción de Secciones Retóricas de Artículos Científicos
Parte 1/4: Configuración, Lectura, Filtrado e Identificación de Artículos
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CORE_DIR = r"C:\Users\nico3\Documents\Proyecto final maestria\core"
OUTPUT_DIR = r"C:\Users\nico3\Documents\Proyecto final maestria\datasets_v2"

MIN_PALABRAS_DOC = 500
MIN_RATIO_TILDES = 0.02
MIN_CHUNK_WORDS = 250
SOFT_MAX_WORDS = 700
HARD_MAX_WORDS = 1000
TARGET_PER_LABEL = 2500
MAX_CHUNKS_PER_DOC_PER_LABEL = 5
RANDOM_SEED = 42

TARGET_LABELS = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ============================================================================
# PATRONES REGEX
# ============================================================================

MESES = (r"enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|"
         r"setiembre|octubre|noviembre|diciembre")

HEADING_PATTERNS = {
    "INTRO": [
        r"introducci[oó]n", r"presentaci[oó]n",
        r"objetivo(?:s)?(?:\s+general(?:es)?|\s+espec[ií]fico(?:s)?)?",
        r"planteamiento(?:\s+del\s+problema)?",
        r"contextualizaci[oó]n", r"justificaci[oó]n(?:\s+del\s+estudio)?",
    ],
    "BACK": [
        r"estado\s+del\s+arte", r"estado\s+de\s+la\s+cuesti[oó]n",
        r"trabajos?\s+relacionados?", r"trabajos?\s+previos?",
        r"antecedentes", r"marco\s+contextual", r"marco\s+referencial",
        r"marco\s+te[oó]rico", r"marco\s+conceptual",
        r"revisi[oó]n\s+de\s+(la\s+)?literatura",
        r"revisi[oó]n\s+bibliogr[aá]fica",
        r"fundamentos?\s+te[oó]ricos?",
        r"fundamentaci[oó]n(?:\s+te[oó]rica)?",
        r"bases?\s+te[oó]ricas?", r"referentes?\s+te[oó]ricos?",
        r"contexto\s+te[oó]rico",
    ],
    "METH": [
        r"metodolog[ií]a", r"m[eé]todo(?:s)?",
        r"materiales?\s+y\s+m[eé]todos?",
        r"dise[nñ]o\s+(experimental|metodol[oó]gico|de\s+investigaci[oó]n)",
        r"procedimiento", r"marco\s+metodol[oó]gico",
        r"propuesta\s+metodol[oó]gica", r"enfoque\s+metodol[oó]gico",
        r"an[aá]lisis\s+estad[ií]stico",
    ],
    "RES": [
        r"resultados?", r"hallazgos?",
        r"an[aá]lisis\s+de\s+(los\s+)?resultados?",
        r"resultados?\s+obtenidos?",
    ],
    "DISC": [
        r"discusi[oó]n", r"discusi[oó]n\s+de\s+(los\s+)?resultados?",
        r"interpretaci[oó]n(?:\s+de\s+(los\s+)?resultados?)?",
        r"an[aá]lisis\s+e\s+interpretaci[oó]n",
    ],
    "CONTR": [
        r"contribuci[oó]n(?:es)?", r"aportes?(?:\s+del\s+trabajo)?",
    ],
    "LIM": [
        r"limitaci[oó]n(?:es)?", r"trabajos?\s+futuros?",
        r"l[ií]neas?\s+futuras?",
        r"limitaci[oó]n(?:es)?\s+y\s+trabajos?\s+futuros?",
    ],
    "CONC": [
        r"conclusi[oó]n(?:es)?", r"consideraci[oó]n(?:es)?\s+finales?",
        r"reflexi[oó]n(?:es)?\s+finales?",
        r"conclusi[oó]n(?:es)?\s+y\s+recomendaciones?",
    ],
}

MIXED_HEADING_PATTERNS_RAW = [
    r"resultados?\s+y\s+discusi[oó]n(?:\s+de\s+resultados?)?",
    r"discusi[oó]n\s+y\s+resultados?",
    r"resultados?\s+y\s+an[aá]lisis",
    r"an[aá]lisis\s+y\s+discusi[oó]n",
]

SKIP_HEADING_PATTERNS_RAW = [
    r"referencias?", r"referencias?\s+bibliogr[aá]ficas?",
    r"bibliograf[ií]a", r"references?",
    r"anexos?", r"ap[eé]ndices?", r"agradecimientos?",
    r"glosario", r"[ií]ndice(?:\s+general|\s+de\s+contenido)?",
    r"tabla\s+de\s+contenido", r"resumen", r"abstract",
    r"palabras\s+clave", r"key\s*words?",
    r"cronograma", r"presupuesto",
    r"contribuci[oó]n\s+de\s+autor[ií]a",
    r"conflictos?\s+de\s+inter[eé]s",
    r"financiaci[oó]n", r"financiamiento",
    r"disponibilidad\s+de\s+datos",
]

ABSTRACT_MARKERS = [r"resumen", r"abstract", r"palabras\s+clave", r"key\s*words?"]

ARTICLE_HEADER_MARKERS = [
    r"resumen\s*[\n:]", r"abstract\s*[\n:]", r"palabras\s+clave",
    r"keywords?", r"recibido[:\s]", r"aceptado[:\s]",
    r"\bdoi\b", r"\bissn\b", r"\bvol\.?\s*\d+", r"\brevista\b",
    r"c[oó]mo\s+citar",
]

DISCARD_MARKERS = [
    r"diario\s+oficial", r"bolet[ií]n\s+oficial", r"ministerio\s+del",
    r"\bedicto\b", r"anuncios?\s+particulares",
    r"di[aá]logo\s+con", r"entrevista\s+a",
    r"entrevistad[oa]", r"entrevistador",
]

LEXICAL_STRONG = {
    "INTRO": [
        r"el\s+objetivo\s+de\s+est[ea]\s+(trabajo|estudio|investigaci[oó]n)",
        r"el\s+prop[oó]sito\s+de\s+est[ea]",
        r"est[ea]\s+(art[ií]culo|trabajo|estudio)\s+presenta",
        r"tiene\s+como\s+objetivo",
        r"en\s+est[ea]\s+(estudio|trabajo)\s+se\s+(analiza|propone|presenta)",
        r"la\s+presente\s+investigaci[oó]n",
        r"el\s+objetivo\s+del\s+presente\s+(trabajo|estudio|art[ií]culo)",
    ],
    "BACK": [
        r"estudios\s+previos\s+(han\s+mostrado|indican|sugieren)",
        r"en\s+la\s+literatura\s+se\s+ha\s+reportado",
        r"investigaciones\s+anteriores",
        r"trabajos\s+previos\s+(han|muestran|reportan)",
        r"la\s+literatura\s+(muestra|reporta|se[nñ]ala)",
        r"se\s+ha\s+documentado\s+que",
    ],
    "METH": [
        r"la\s+muestra\s+estuvo\s+compuesta",
        r"los\s+datos\s+fueron\s+recolectados",
        r"se\s+aplic[oó]\s+un\s+dise[nñ]o",
        r"la\s+investigaci[oó]n\s+(es|fue)\s+de\s+tipo",
        r"la\s+poblaci[oó]n\s+(de\s+estudio|estudiada|objetivo)",
        r"los\s+participantes\s+fueron",
        r"se\s+recolectaron\s+los\s+datos",
        r"el\s+procedimiento\s+consisti[oó]\s+en",
    ],
    "RES": [
        r"los\s+resultados\s+(muestran|indican|evidencian|revelan)",
        r"se\s+(encontr[oó]|hall[oó]|observ[oó])\s+que",
        r"los\s+datos\s+indican",
        r"el\s+an[aá]lisis\s+(mostr[oó]|revel[oó]|indic[oó])",
        r"se\s+obtuv(?:o|ieron)\s+(un|los|las)",
        r"se\s+evidenci[oó]\s+que",
    ],
    "DISC": [
        r"estos\s+resultados\s+sugieren",
        r"en\s+comparaci[oó]n\s+con\s+(estudios|trabajos)\s+previos",
        r"una\s+posible\s+explicaci[oó]n",
        r"este\s+hallazgo\s+es\s+consistente\s+con",
        r"esto\s+(podr[ií]a|puede)\s+explicarse",
        r"implica\s+que",
    ],
    "CONC": [
        r"en\s+conclusi[oó]n", r"se\s+concluye\s+que",
        r"a\s+modo\s+de\s+cierre", r"en\s+s[ií]ntesis",
        r"en\s+resumen", r"en\s+definitiva",
    ],
    "CONTR": [
        r"la\s+principal\s+contribuci[oó]n",
        r"nuestra\s+contribuci[oó]n",
        r"el\s+aporte\s+de\s+est[ea]\s+investigaci[oó]n",
        r"est[ea]\s+trabajo\s+aporta",
        r"proponemos\s+un", r"presentamos\s+un",
        r"se\s+introduce\s+un\s+(nuevo|novedoso)",
        r"hemos\s+desarrollado",
    ],
    "LIM": [
        r"una\s+limitaci[oó]n\s+de\s+est[ea]\s+estudio",
        r"entre\s+las\s+limitaciones",
        r"no\s+fue\s+posible\s+(evaluar|analizar|medir)",
        r"futuras\s+investigaciones\s+deber[ií]an",
        r"l[ií]neas?\s+futuras?",
    ],
}

LEXICAL_WEAK = {
    "INTRO": [
        r"el\s+presente\s+(trabajo|estudio|art[ií]culo)",
        r"se\s+busca\s+(analizar|evaluar|determinar|identificar)",
    ],
    "BACK": [
        r"seg[uú]n\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\-]+",
        r"de\s+acuerdo\s+con\s+[A-ZÁÉÍÓÚÑ]",
        r"como\s+se[nñ]alan?\s+[A-ZÁÉÍÓÚÑ]",
        r"diversos\s+autores?\s+(han|coinciden|sostienen)",
    ],
    "METH": [
        r"se\s+utiliz[oó]", r"se\s+emple[oó]\s+el\s+m[eé]todo",
        r"para\s+el\s+an[aá]lisis\s+se\s+utiliz[oó]",
    ],
    "RES": [
        r"la\s+tabla\s+\d+\s+(muestra|presenta)",
        r"como\s+se\s+(observa|aprecia|muestra)\s+en",
    ],
    "DISC": [
        r"cabe\s+se[nñ]alar\s+que",
        r"resulta\s+interesante\s+que", r"da\s+cuenta\s+de",
    ],
    "CONC": [
        r"a\s+partir\s+de\s+lo\s+expuesto",
        r"los\s+hallazgos\s+de\s+est[ea]\s+estudio",
    ],
    "CONTR": [
        r"est[ea]\s+trabajo\s+contribuye",
        r"aporta\s+una\s+perspectiva", r"aporte(?:s)?",
    ],
    "LIM": [
        r"se\s+recomienda\s+(para|en)\s+futuros?",
        r"queda\s+pendiente",
        r"ser[ií]a\s+conveniente\s+(explorar|investigar|analizar)",
    ],
}

RESCUE_CONTR = [
    r"este\s+trabajo\s+propone", r"este\s+art[ií]culo\s+presenta",
    r"presentamos", r"proponemos", r"introducimos",
    r"se\s+propone\s+un\s+nuevo", r"proponemos\s+un\s+nuevo",
    r"la\s+principal\s+contribuci[oó]n",
    r"las\s+principales\s+contribuciones?",
    r"nuestra\s+contribuci[oó]n",
    r"desarrollamos\s+un\s+m[eé]todo", r"aportamos",
    r"este\s+trabajo\s+aporta", r"aporte\s+principal",
    r"a\s+diferencia\s+de\s+trabajos?\s+previos?",
    r"novedad", r"novedoso", r"nuevo\s+enfoque",
    r"enfoque\s+novedoso", r"avance",
    r"contribuci[oó]n\s+original", r"por\s+primera\s+vez",
    r"primer\s+estudio\s+que", r"primer\s+trabajo\s+en",
]

RESCUE_LIM = [
    r"limitaci[oó]n(?:es)?", r"una\s+limitaci[oó]n",
    r"principales\s+limitaciones?", r"limitante",
    r"restricci[oó]n(?:es)?\s+del\s+estudio",
    r"entre\s+las\s+limitaciones",
    r"no\s+se\s+consider[oó]", r"no\s+fue\s+posible", r"no\s+pudo",
    r"futuras\s+investigaciones",
    r"trabajos\s+futuros?\s+deber[ií]an",
    r"futuros?\s+estudios", r"sesgo(?:s)?",
    r"tama[nñ]o\s+de\s+la\s+muestra", r"tama[nñ]o\s+muestral",
    r"alcance\s+del\s+estudio", r"alcance\s+limitado",
    r"no\s+se\s+generaliza", r"generalizaci[oó]n\s+limitada",
    r"fuera\s+del\s+alcance",
    r"se\s+recomienda\s+realizar\s+futuras",
    r"posibles\s+fuentes\s+de\s+error",
]

POSITION_ZONES = {
    "INTRO": (0.00, 0.15), "BACK": (0.08, 0.35),
    "METH": (0.18, 0.58), "RES": (0.34, 0.76),
    "DISC": (0.56, 0.88), "CONC": (0.88, 1.00),
}

# --- Compilación ---

def _compile_heading(patterns):
    return [re.compile(rf"^(?:{p})[\s\.:;\-]*$", re.IGNORECASE) for p in patterns]

def _compile_search(patterns):
    return [re.compile(p, re.IGNORECASE) for p in patterns]

HEADING_COMPILED = {lab: _compile_heading(pats) for lab, pats in HEADING_PATTERNS.items()}
MIXED_COMPILED = _compile_heading(MIXED_HEADING_PATTERNS_RAW)
SKIP_COMPILED = _compile_heading(SKIP_HEADING_PATTERNS_RAW)
ABSTRACT_COMPILED = _compile_heading(ABSTRACT_MARKERS)

LEXICAL_STRONG_C = {lab: _compile_search(pats) for lab, pats in LEXICAL_STRONG.items()}
LEXICAL_WEAK_C = {lab: _compile_search(pats) for lab, pats in LEXICAL_WEAK.items()}
RESCUE_CONTR_C = _compile_search(RESCUE_CONTR)
RESCUE_LIM_C = _compile_search(RESCUE_LIM)

DISCARD_COMPILED = _compile_search(DISCARD_MARKERS)
EDITORIAL_COMPILED = _compile_search([
    rf"\b(?:revista|journal|vol\.?|n[oº]\.?|issn|isbn|doi)\b",
    rf"\b(?:{MESES})\b.*\b\d{{4}}\b",
    r"\brecibido\b|\baceptado\b|\bpublicado\b",
])
CAPTION_COMPILED = _compile_search([
    r"^(?:tabla|figura|cuadro|gr[aá]fico|foto|anexo)\s*[\dIVXLCM\-\.]*\b",
    r"^fuente\s*:",
])
REF_LINE_COMPILED = _compile_search([
    r"https?://|doi[:\s]|doi\.org|www\.",
    r"recuperado\s+de|disponible\s+en",
])

# ============================================================================
# MÓDULO 1: LECTURA Y FILTRADO
# ============================================================================

def leer_documento(ruta):
    for enc in ["utf-8", "latin-1"]:
        try:
            with open(ruta, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(ruta, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def norm(linea):
    return re.sub(r"\s+", " ", linea.strip())


def contar_palabras(texto):
    return len(texto.split())


def filtro_rapido(texto):
    palabras = texto.split()
    n = len(palabras)
    if n < MIN_PALABRAS_DOC:
        return False, "muy_corto"
    con_tilde = sum(1 for p in palabras if re.search(r"[áéíóúñüÁÉÍÓÚÑÜ]", p))
    if con_tilde / n < MIN_RATIO_TILDES:
        return False, "no_espanol"
    return True, "ok"


def deduplicar(documentos):
    vistos = set()
    unicos = []
    dupes = 0
    for ruta, texto in documentos:
        h = hashlib.md5(" ".join(texto.split()[:500]).encode()).hexdigest()
        if h in vistos:
            dupes += 1
            continue
        vistos.add(h)
        unicos.append((ruta, texto))
    return unicos, dupes


# ============================================================================
# MÓDULO 2: IDENTIFICACIÓN DE ARTÍCULOS
# ============================================================================

def split_heading_prefix(linea):
    l = norm(linea)
    m = re.match(
        r"^(?P<prefix>(?:\d+(?:\.\d+){0,2}|[IVXLCDM]+))[.\)\-:]{0,3}\s+(?P<body>.+)$",
        l, re.IGNORECASE
    )
    if not m:
        return l, False
    return m.group("body").strip(), True


def normalizar_heading(texto):
    t = norm(texto)
    t = re.sub(r"^[\W_]+|[\W_]+$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def clasificar_linea_heading(linea, idx, lineas):
    l = norm(linea)
    if not l or len(l) > 120:
        return None
    if re.match(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ].*\d{4}", l):
        return None
    if re.search(r"https?://|doi[:\s]|www\.", l, re.IGNORECASE):
        return None

    cuerpo, tenia_numero = split_heading_prefix(l)
    cuerpo_n = normalizar_heading(cuerpo)
    if not cuerpo_n or len(cuerpo_n.split()) > 12:
        return None

    # Skip heading?
    if any(p.search(cuerpo_n) for p in SKIP_COMPILED):
        return {"tipo": "skip", "encabezado": cuerpo_n, "etiqueta": None}

    # Estándar?
    for lab, pats in HEADING_COMPILED.items():
        if any(p.search(cuerpo_n) for p in pats):
            return {"tipo": "estandar", "encabezado": cuerpo_n, "etiqueta": lab}

    # Mixto?
    if any(p.search(cuerpo_n) for p in MIXED_COMPILED):
        return {"tipo": "mixto", "encabezado": cuerpo_n, "etiqueta": None, "candidatas": ["RES", "DISC"]}

    # Genérico (mayúsculas, aislado, corto)?
    if len(cuerpo_n.split()) <= 8:
        previo = lineas[idx-1].strip() if idx > 0 else ""
        siguiente = lineas[idx+1].strip() if idx+1 < len(lineas) else ""
        aislado = not previo or not siguiente
        es_mayus = cuerpo_n == cuerpo_n.upper() and re.search(r"[A-ZÁÉÍÓÚÑ]{3,}", cuerpo_n)
        es_titulo = sum(1 for p in cuerpo_n.split() if p[0:1].isupper()) >= max(1, len(cuerpo_n.split())-1)
        if aislado and (es_mayus or es_titulo or tenia_numero):
            if not re.search(r"\b(universidad|hospital|facultad|departamento|autor|correo|e-?mail)\b", cuerpo_n, re.IGNORECASE):
                if not re.search(r"[.;,:!?]$", cuerpo_n):
                    return {"tipo": "generico", "encabezado": cuerpo_n, "etiqueta": None}

    return None


def detectar_encabezados(texto):
    lineas = texto.splitlines()
    encabezados = []
    for i, linea in enumerate(lineas):
        info = clasificar_linea_heading(linea, i, lineas)
        if info:
            info["linea"] = i
            encabezados.append(info)
    return encabezados


def es_articulo_valido(texto, encabezados):
    texto_lower = texto.lower()
    cabecera = "\n".join(texto.splitlines()[:140])[:5000].lower()

    # Señales de descarte
    for p in DISCARD_COMPILED:
        if p.search(cabecera):
            return False

    # Libro largo
    if contar_palabras(texto) > 30000:
        if re.search(r"\bisbn\b|cap[ií]tulo\s+[IVXivx\d]+", cabecera, re.IGNORECASE):
            return False

    # Señales positivas
    score = 0
    estandar = [e for e in encabezados if e["tipo"] == "estandar"]
    etiquetas = set(e["etiqueta"] for e in estandar if e["etiqueta"])

    if len(estandar) >= 2:
        score += 2
    elif len(estandar) >= 1:
        score += 1

    if len(etiquetas) >= 3:
        score += 1

    for patron in ARTICLE_HEADER_MARKERS:
        if re.search(patron, cabecera, re.IGNORECASE):
            score += 0.5

    mixtos = [e for e in encabezados if e["tipo"] == "mixto"]
    if mixtos:
        score += 1

    genericos = [e for e in encabezados if e["tipo"] == "generico"]
    if len(genericos) >= 3:
        score += 0.5

    return score >= 2


# ============================================================================
# MÓDULO 3: ZONIFICACIÓN
# ============================================================================


def parece_referencia(linea):
    l = norm(linea)
    if not l:
        return False
    if re.match(r"^\d+\.\s+[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\-]+.*\d{4}", l):
        return True
    if any(p.search(l) for p in REF_LINE_COMPILED):
        return True
    return False


def parece_editorial(linea):
    l = norm(linea)
    if not l:
        return False
    if re.fullmatch(r"\d{1,4}", l):
        return True
    if re.fullmatch(r"[ivxlcdm]{1,8}", l.lower()):
        return True
    if any(p.search(l) for p in EDITORIAL_COMPILED) and len(l.split()) <= 25:
        return True
    return False


def parece_caption(linea):
    l = norm(linea)
    return bool(l) and any(p.search(l) for p in CAPTION_COMPILED)


def es_linea_tabular(linea):
    l = norm(linea)
    if not l:
        return False
    tokens = l.split()
    if len(tokens) >= 4:
        numericos = sum(1 for t in tokens if re.search(r"\d", t))
        cortos = sum(1 for t in tokens if len(t) <= 4)
        if numericos >= 3 and cortos / max(len(tokens), 1) >= 0.5:
            return True
    if re.search(r"\b[pP]\s*[<=>\u2264\u2265]\s*0\.\d+", l):
        return True
    return False


def detectar_repetidas(lineas):
    contador = Counter()
    for linea in lineas:
        l = norm(linea)
        if len(l) < 12 or len(l) > 120 or len(l.split()) > 18:
            continue
        if parece_referencia(l):
            continue
        contador[l.lower()] += 1
    return {l for l, c in contador.items() if c >= 3 and len(l.split()) <= 10}


def encontrar_inicio_cuerpo(lineas, encabezados):
    """Encuentra la primera línea del cuerpo (después de abstract/front-matter)."""
    # Buscar primer encabezado skip tipo abstract
    primer_abstract = None
    primer_estandar = None
    limite = min(250, len(lineas))

    for enc in encabezados:
        if enc["linea"] >= limite:
            break
        if enc["tipo"] == "skip":
            h = normalizar_heading(enc["encabezado"])
            if any(p.search(h) for p in ABSTRACT_COMPILED):
                primer_abstract = enc["linea"]
                break
        if enc["tipo"] == "estandar" and primer_estandar is None:
            primer_estandar = enc["linea"]
            break
        if enc["tipo"] == "mixto" and primer_estandar is None:
            primer_estandar = enc["linea"]
            break

    # Si hay abstract, buscar dónde termina
    if primer_abstract is not None:
        # Buscar siguiente heading estándar/mixto después del abstract
        for enc in encabezados:
            if enc["linea"] <= primer_abstract:
                continue
            if enc["tipo"] in ("estandar", "mixto"):
                return enc["linea"]
            if enc["tipo"] == "generico":
                return enc["linea"]
        # Si no encontramos nada después, buscar por palabras clave + salto
        saw_keywords = False
        words_seen = 0
        for i in range(primer_abstract + 1, min(len(lineas), primer_abstract + 200)):
            l = norm(lineas[i])
            if re.search(r"palabras\s+clave|key\s*words?", l, re.IGNORECASE):
                saw_keywords = True
            if l:
                words_seen += contar_palabras(l)
            if saw_keywords and not l and words_seen >= 100:
                # Buscar siguiente línea no vacía con contenido sustancial
                for j in range(i + 1, min(len(lineas), i + 10)):
                    if lineas[j].strip() and contar_palabras(norm(lineas[j])) >= 20:
                        return j
            if words_seen >= 300 and not l:
                for j in range(i + 1, min(len(lineas), i + 10)):
                    if lineas[j].strip() and contar_palabras(norm(lineas[j])) >= 20:
                        return j
        return primer_abstract

    if primer_estandar is not None:
        return primer_estandar

    return 0


def encontrar_fin_cuerpo(lineas, encabezados):
    """Encuentra la línea donde empiezan las referencias (fin del cuerpo)."""
    total = len(lineas)

    # Buscar headings skip desde el final
    for enc in reversed(encabezados):
        if enc["tipo"] != "skip":
            continue
        h = normalizar_heading(enc["encabezado"]).lower()
        if any(p.search(h) for p in _compile_heading(
            [r"referencias?", r"referencias?\s+bibliogr[aá]ficas?",
             r"bibliograf[ií]a", r"references?", r"anexos?", r"ap[eé]ndices?"]
        )):
            idx = enc["linea"]
            if idx > total * 0.4:
                return idx

    # Fallback: buscar patrón de referencias desde el final
    for i in range(total - 1, max(0, total - 400), -1):
        l = norm(lineas[i]).lower()
        if re.match(r"^(\d+[\.\)]\s*)?(referencias?|bibliograf[ií]a|references)\s*$", l):
            if i > total * 0.4:
                return i

    return total


def limpiar_cuerpo(lineas, repetidas):
    """Elimina ruido del cuerpo: editoriales, captions, tabulares, repetidas."""
    limpias = []
    last_blank = True

    for linea in lineas:
        l = norm(linea)

        if not l:
            if not last_blank:
                limpias.append("")
            last_blank = True
            continue

        if l.lower() in repetidas:
            continue
        if parece_editorial(l):
            continue
        if parece_referencia(l) and len(l.split()) <= 20:
            continue
        if parece_caption(l) and contar_palabras(l) <= 18:
            continue
        if es_linea_tabular(l) and contar_palabras(l) <= 20:
            continue

        limpias.append(linea.rstrip())
        last_blank = False

    # Trim blanks
    while limpias and not limpias[0].strip():
        limpias.pop(0)
    while limpias and not limpias[-1].strip():
        limpias.pop()

    return limpias


def zonificar_documento(texto, encabezados):
    """Retorna las líneas del cuerpo limpio del documento."""
    lineas = texto.splitlines()
    inicio = encontrar_inicio_cuerpo(lineas, encabezados)
    fin = encontrar_fin_cuerpo(lineas, encabezados)

    if fin <= inicio:
        fin = len(lineas)

    cuerpo_bruto = lineas[inicio:fin]
    repetidas = detectar_repetidas(lineas)
    cuerpo_limpio = limpiar_cuerpo(cuerpo_bruto, repetidas)

    return cuerpo_limpio, inicio, fin, len(lineas)


# ============================================================================
# MÓDULO 4: SEGMENTACIÓN
# ============================================================================


def segmentar_secciones(lineas_cuerpo):
    """Divide el cuerpo en secciones delimitadas por encabezados."""
    secciones = []
    actual = {
        "encabezado": None, "tipo_heading": None,
        "etiqueta": None, "candidatas": [],
        "lineas": [], "linea_inicio": 0,
    }

    for idx, linea in enumerate(lineas_cuerpo):
        info = clasificar_linea_heading(linea, idx, lineas_cuerpo) if linea.strip() else None

        if info and info["tipo"] in ("estandar", "mixto", "generico"):
            # Guardar sección anterior si tiene contenido
            if actual["lineas"] or actual["encabezado"]:
                secciones.append(actual)

            # Heredar etiqueta para genéricos
            etiqueta_heredada = actual["etiqueta"] if info["tipo"] == "generico" else None
            candidatas_heredadas = list(actual["candidatas"]) if info["tipo"] == "generico" else []

            actual = {
                "encabezado": info["encabezado"],
                "tipo_heading": info["tipo"],
                "etiqueta": info.get("etiqueta") or etiqueta_heredada,
                "candidatas": info.get("candidatas", []) or candidatas_heredadas,
                "lineas": [],
                "linea_inicio": idx,
            }
            continue

        if info and info["tipo"] == "skip":
            # Guardar sección anterior y saltar hasta siguiente heading
            if actual["lineas"] or actual["encabezado"]:
                secciones.append(actual)
            actual = {
                "encabezado": None, "tipo_heading": "skip",
                "etiqueta": None, "candidatas": [],
                "lineas": [], "linea_inicio": idx,
            }
            continue

        # Solo acumular si no estamos en skip
        if actual["tipo_heading"] != "skip":
            actual["lineas"].append(linea)

    if actual["lineas"] or actual["encabezado"]:
        if actual["tipo_heading"] != "skip":
            secciones.append(actual)

    return [s for s in secciones if s["lineas"]]


def normalizar_parrafo(lineas):
    """Une líneas de un párrafo, resolviendo guiones y espacios."""
    partes = []
    for linea in lineas:
        l = norm(linea)
        if not l:
            continue
        if partes and partes[-1].endswith("-") and l[:1].islower():
            partes[-1] = partes[-1][:-1] + l
        else:
            partes.append(l)
    texto = " ".join(partes)
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s+([,.;:!?])", r"\1", texto)
    return texto.strip()


def es_parrafo_ruidoso(texto):
    """Detecta párrafos que son ruido (tablas, refs, metadata)."""
    if not texto:
        return True
    if parece_editorial(texto):
        return True
    if parece_referencia(texto) and contar_palabras(texto) <= 60:
        return True
    # Muchas citas bibliográficas
    years = len(re.findall(r"\b(?:19|20)\d{2}\b", texto))
    autores = len(re.findall(r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ\-]+,\s*(?:[A-ZÁÉÍÓÚÑ]\.)", texto))
    et_al = len(re.findall(r"\bet\s+al\.?", texto, re.IGNORECASE))
    if years >= 4 and (autores >= 3 or et_al >= 2):
        return True
    # Muy tabular
    tokens = texto.split()
    if len(tokens) >= 12:
        numericos = sum(1 for t in tokens if re.search(r"\d", t))
        cortos = sum(1 for t in tokens if len(re.sub(r"\W+", "", t)) <= 4)
        if numericos >= 6 and numericos / len(tokens) >= 0.14 and cortos / len(tokens) >= 0.45:
            return True
    # Caption suelto
    if re.search(r"^(?:foto|figura|tabla|cuadro)\s+\d+", texto, re.IGNORECASE):
        return True
    return False


def extraer_parrafos(lineas):
    """Agrupa líneas consecutivas no-vacías en párrafos normalizados."""
    parrafos = []
    buffer = []

    def flush():
        if not buffer:
            return
        texto = normalizar_parrafo(buffer)
        if texto and not es_parrafo_ruidoso(texto) and contar_palabras(texto) >= 5:
            parrafos.append(texto)

    for linea in lineas:
        l = norm(linea)
        # Líneas ruidosas individuales cortan el párrafo
        if l and (parece_caption(l) or es_linea_tabular(l) or parece_editorial(l)):
            flush()
            buffer = []
            continue

        if linea.strip():
            buffer.append(linea)
        else:
            flush()
            buffer = []

    flush()
    return parrafos


def dividir_por_oraciones(texto):
    """Divide texto en oraciones para subdivisión de párrafos largos."""
    partes = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9\(])", texto)
    if len(partes) <= 1:
        partes = re.split(r"\s*;\s+", texto)
    return [p.strip() for p in partes if contar_palabras(p) >= 5]


def subdividir_largo(texto):
    """Subdivide un texto > HARD_MAX en segmentos de SOFT_MAX."""
    if contar_palabras(texto) <= HARD_MAX_WORDS:
        return [texto]

    oraciones = dividir_por_oraciones(texto)
    if not oraciones:
        # Fallback: partir por ventana de palabras
        palabras = texto.split()
        segs = []
        for i in range(0, len(palabras), SOFT_MAX_WORDS):
            seg = " ".join(palabras[i:i + SOFT_MAX_WORDS]).strip()
            if seg:
                segs.append(seg)
        return segs

    segmentos = []
    buf = []
    buf_n = 0

    for oracion in oraciones:
        n = contar_palabras(oracion)
        if n > HARD_MAX_WORDS:
            if buf:
                segmentos.append(" ".join(buf))
                buf, buf_n = [], 0
            palabras = oracion.split()
            for i in range(0, len(palabras), SOFT_MAX_WORDS):
                segmentos.append(" ".join(palabras[i:i + SOFT_MAX_WORDS]))
            continue

        if buf_n and buf_n + n > SOFT_MAX_WORDS and buf_n >= MIN_CHUNK_WORDS:
            segmentos.append(" ".join(buf))
            buf, buf_n = [oracion], n
            continue

        if buf_n and buf_n + n > HARD_MAX_WORDS:
            segmentos.append(" ".join(buf))
            buf, buf_n = [oracion], n
            continue

        buf.append(oracion)
        buf_n += n

    if buf:
        ultimo = " ".join(buf)
        if segmentos and contar_palabras(ultimo) < MIN_CHUNK_WORDS:
            candidato = segmentos[-1] + " " + ultimo
            if contar_palabras(candidato) <= HARD_MAX_WORDS:
                segmentos[-1] = candidato
            else:
                segmentos.append(ultimo)
        else:
            segmentos.append(ultimo)

    return [s for s in segmentos if contar_palabras(s) >= 5]


def construir_chunks(parrafos, seccion, total_lineas_cuerpo):
    """Construye chunks de 250-1000 palabras a partir de párrafos."""
    if not parrafos:
        return []

    # Primero subdividir párrafos muy largos
    parrafos_sub = []
    for p in parrafos:
        parrafos_sub.extend(subdividir_largo(p))

    chunks = []
    buf = []
    buf_n = 0

    for parrafo in parrafos_sub:
        n = contar_palabras(parrafo)

        # Si buffer lleno y agregar este excede soft_max
        if buf and buf_n >= MIN_CHUNK_WORDS and buf_n + n > SOFT_MAX_WORDS:
            chunks.append("\n\n".join(buf))
            buf, buf_n = [parrafo], n
            continue

        # Si agregar excede hard_max
        if buf and buf_n + n > HARD_MAX_WORDS:
            chunks.append("\n\n".join(buf))
            buf, buf_n = [parrafo], n
            continue

        buf.append(parrafo)
        buf_n += n

    if buf:
        ultimo = "\n\n".join(buf)
        if chunks and contar_palabras(ultimo) < MIN_CHUNK_WORDS:
            candidato = chunks[-1] + "\n\n" + ultimo
            if contar_palabras(candidato) <= HARD_MAX_WORDS:
                chunks[-1] = candidato
            elif contar_palabras(ultimo) >= 80:
                chunks.append(ultimo)
        elif contar_palabras(ultimo) >= 80:
            chunks.append(ultimo)

    # Construir objetos chunk con metadata
    resultado = []
    for i, texto in enumerate(chunks):
        n_palabras = contar_palabras(texto)
        if n_palabras < MIN_CHUNK_WORDS:
            continue
        # Posición relativa estimada
        pos = (seccion["linea_inicio"] + i * 10) / max(total_lineas_cuerpo, 1)
        pos = min(1.0, pos)

        resultado.append({
            "texto": texto,
            "num_palabras": n_palabras,
            "posicion_relativa": round(pos, 4),
            "encabezado_seccion": seccion.get("encabezado"),
            "tipo_heading": seccion.get("tipo_heading"),
            "etiqueta_heading": seccion.get("etiqueta"),
            "candidatas_heading": seccion.get("candidatas", []),
        })

    return resultado


def procesar_secciones(lineas_cuerpo, total_lineas_doc):
    """Pipeline completo: secciones → párrafos → chunks."""
    secciones = segmentar_secciones(lineas_cuerpo)
    todos_chunks = []

    for sec_idx, seccion in enumerate(secciones):
        parrafos = extraer_parrafos(seccion["lineas"])
        chunks = construir_chunks(parrafos, seccion, len(lineas_cuerpo))
        for chunk_idx, chunk in enumerate(chunks):
            chunk["indice_seccion"] = sec_idx
            todos_chunks.append(chunk)

    return todos_chunks


# ============================================================================
# MÓDULO 5: ETIQUETADO
# ============================================================================


def detectar_patrones(texto):
    """Detecta patrones léxicos fuertes y débiles por etiqueta."""
    scores = {}
    activados = []

    for lab in TARGET_LABELS:
        s_count = 0
        w_count = 0
        for p in LEXICAL_STRONG_C.get(lab, []):
            if p.search(texto):
                s_count += 1
                activados.append(f"strong::{lab}::{p.pattern[:40]}")
        for p in LEXICAL_WEAK_C.get(lab, []):
            if p.search(texto):
                w_count += 1
                activados.append(f"weak::{lab}::{p.pattern[:40]}")
        scores[lab] = {"strong": s_count, "weak": w_count}

    return scores, activados


def posicion_apoya(etiqueta, posicion):
    """True si la posición del chunk cae en la zona esperada de la etiqueta."""
    if etiqueta in POSITION_ZONES:
        lo, hi = POSITION_ZONES[etiqueta]
        return lo <= posicion <= hi
    return False


def puntuar_etiqueta(lab, chunk, evidencia):
    """Calcula score combinado para una etiqueta candidata."""
    s = evidencia["strong"]
    w = evidencia["weak"]
    score = s * 2.5 + w * 0.75
    fuente = []

    h_type = chunk.get("tipo_heading")
    h_lab = chunk.get("etiqueta_heading")
    h_cands = chunk.get("candidatas_heading", [])

    # Bonus por encabezado
    if h_type == "estandar" and h_lab == lab:
        score += 4.5
        fuente.append("encabezado")
    elif h_type == "mixto" and lab in h_cands:
        score += 2.0
        fuente.append("encabezado_mixto")
    elif h_type == "generico" and h_lab == lab:
        score += 1.2
        fuente.append("contexto_generico")

    # Bonus posicional
    pos = chunk.get("posicion_relativa", 0.5)
    if (s or w or "encabezado" in fuente) and posicion_apoya(lab, pos):
        score += 0.5
        fuente.append("posicion")

    return {"score": score, "strong": s, "weak": w, "fuente": fuente}


def etiquetar_chunk(chunk):
    """Asigna etiqueta principal, secundaria, confianza y fuente."""
    texto = chunk["texto"]
    h_type = chunk.get("tipo_heading")
    h_lab = chunk.get("etiqueta_heading")
    pos = chunk.get("posicion_relativa", 0.5)

    evidencia, activados = detectar_patrones(texto)

    # Puntuar todas las etiquetas
    puntuaciones = []
    for lab in TARGET_LABELS:
        p = puntuar_etiqueta(lab, chunk, evidencia[lab])
        puntuaciones.append({"label": lab, **p})

    puntuaciones.sort(key=lambda x: (x["score"], x["strong"], x["weak"]), reverse=True)
    top = puntuaciones[0]

    etiqueta = None
    secundaria = None
    fuente = "sin_asignacion"
    confianza = "ninguna"

    # Caso 1: Encabezado estándar
    if h_type == "estandar" and h_lab:
        etiqueta = h_lab
        h_score = next(p for p in puntuaciones if p["label"] == h_lab)
        has_patterns = h_score["strong"] + h_score["weak"] > 0
        fuente = "encabezado+patron" if has_patterns else "encabezado"
        confianza = "alta"

    # Caso 2: Encabezado mixto (RES+DISC)
    elif h_type == "mixto":
        cands = [p for p in puntuaciones if p["label"] in chunk.get("candidatas_heading", [])]
        if len(cands) >= 2:
            mejor, otro = cands[0], cands[1]
            if mejor["score"] - otro["score"] >= 0.5 or mejor["strong"] > otro["strong"]:
                etiqueta = mejor["label"]
                secundaria = otro["label"]
                fuente = "encabezado_mixto+patron" if mejor["strong"] else "encabezado_mixto"
                confianza = "media"

    # Caso 3: Solo patrones léxicos (sin encabezado directo)
    if not etiqueta:
        elegibles = []
        for p in puntuaciones:
            if p["label"] in ("CONTR", "LIM"):
                continue  # Se manejan en rescate
            if p["strong"] >= 1 and p["score"] >= 2.4:
                elegibles.append(p)
            elif "contexto_generico" in p["fuente"] and p["strong"] >= 1 and p["score"] >= 2.0:
                elegibles.append(p)
            elif p["label"] in ("INTRO", "METH", "RES", "CONC") and p["weak"] >= 2 and p["score"] >= 1.5:
                elegibles.append(p)

        if elegibles:
            mejor = elegibles[0]
            rival = elegibles[1] if len(elegibles) > 1 else None
            if not rival or mejor["score"] - rival["score"] >= 0.75 or mejor["strong"] > rival.get("strong", 0):
                etiqueta = mejor["label"]
                secundaria = rival["label"] if rival and rival["score"] >= 2.0 else None
                if mejor["strong"] >= 2:
                    fuente, confianza = "patron", "alta"
                elif mejor["strong"] >= 1:
                    fuente = "patron+posicion" if "posicion" in mejor["fuente"] else "patron"
                    confianza = "media"
                else:
                    fuente, confianza = "patron", "baja"

    # Caso 4: Posición extrema como fallback
    if not etiqueta:
        if pos <= 0.06:
            etiqueta, fuente, confianza = "INTRO", "posicion_extrema", "baja"
        elif pos >= 0.94:
            etiqueta, fuente, confianza = "CONC", "posicion_extrema", "baja"

    # Caso 5: CONTR/LIM por dominancia de patrones
    for especial in ("CONTR", "LIM"):
        ev = evidencia[especial]
        if ev["strong"] >= 1 and ev["strong"] + ev["weak"] >= 2:
            if not etiqueta:
                etiqueta = especial
                fuente, confianza = "patron_dominante", "media"
            elif not secundaria:
                secundaria = especial

    if not etiqueta:
        etiqueta = "sin_asignacion"
        fuente, confianza = "sin_asignacion", "ninguna"

    chunk["etiqueta"] = etiqueta
    chunk["etiqueta_secundaria"] = secundaria
    chunk["confianza"] = confianza
    chunk["fuente_etiqueta"] = fuente
    chunk["patrones_activados"] = activados
    return chunk


# --- Rescate CONTR / LIM ---

def dividir_en_oraciones(texto):
    """Divide texto en oraciones."""
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        return []
    partes = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ0-9\(])", texto)
    if len(partes) <= 1:
        partes = re.split(r"\s*;\s+", texto)
    return [p.strip() for p in partes if contar_palabras(p) >= 5]


def rescate_nivel1(chunks):
    """Nivel 1: Reetiqueta chunks por densidad de oraciones CONTR/LIM."""
    origenes_validos = {"sin_asignacion", "INTRO", "DISC", "CONC", "CONTR", "LIM", "BACK"}

    for chunk in chunks:
        if chunk["etiqueta"] not in origenes_validos:
            continue

        oraciones = dividir_en_oraciones(chunk["texto"])
        if not oraciones:
            continue

        for lab, patrones in [("CONTR", RESCUE_CONTR_C), ("LIM", RESCUE_LIM_C)]:
            if chunk["etiqueta"] == lab:
                continue

            hits = 0
            patterns_found = set()
            for oracion in oraciones:
                for p in patrones:
                    if p.search(oracion):
                        hits += 1
                        patterns_found.add(p.pattern[:40])
                        break  # una vez por oración

            if len(patterns_found) < 2:
                continue

            pct = hits / len(oraciones)
            if pct < 0.25:
                continue

            # Reetiquetamos
            chunk["etiqueta_original"] = chunk["etiqueta"]
            chunk["etiqueta"] = lab
            chunk["fuente_etiqueta"] = "rescate_densidad"
            chunk["confianza"] = "alta" if pct >= 0.40 and len(patterns_found) >= 3 else "media"
            chunk["pct_rescate"] = round(pct, 4)
            chunk["patrones_rescate"] = sorted(patterns_found)
            break  # solo un rescate por chunk

    return chunks


def rescate_nivel2(chunks, target_contr, target_lim):
    """Nivel 2: Ventana deslizante para extraer pasajes CONTR/LIM embebidos."""
    nuevos = []

    for chunk in chunks:
        if chunk["etiqueta"] in ("CONTR", "LIM"):
            continue

        oraciones = dividir_en_oraciones(chunk["texto"])
        if len(oraciones) < 3:
            continue

        for lab, patrones, target_needed in [
            ("CONTR", RESCUE_CONTR_C, target_contr),
            ("LIM", RESCUE_LIM_C, target_lim),
        ]:
            if target_needed <= 0:
                continue

            # Encontrar secuencias de oraciones con hits
            hits_mask = []
            for oracion in oraciones:
                hit = any(p.search(oracion) for p in patrones)
                hits_mask.append(hit)

            # Buscar ventanas de 3+ oraciones consecutivas con densidad >= 50%
            best_start, best_end, best_density = -1, -1, 0
            for start in range(len(oraciones)):
                for end in range(start + 3, min(start + 12, len(oraciones) + 1)):
                    window = hits_mask[start:end]
                    density = sum(window) / len(window)
                    if density >= 0.50 and density > best_density:
                        best_start, best_end, best_density = start, end, density

            if best_start < 0:
                continue

            pasaje = " ".join(oraciones[best_start:best_end])
            n_palabras = contar_palabras(pasaje)

            if n_palabras < MIN_CHUNK_WORDS or n_palabras > HARD_MAX_WORDS:
                continue

            patterns_found = set()
            for oracion in oraciones[best_start:best_end]:
                for p in patrones:
                    if p.search(oracion):
                        patterns_found.add(p.pattern[:40])

            nuevo = {
                "texto": pasaje,
                "num_palabras": n_palabras,
                "posicion_relativa": chunk.get("posicion_relativa", 0.5),
                "encabezado_seccion": chunk.get("encabezado_seccion"),
                "tipo_heading": "ventana_rescate",
                "etiqueta_heading": None,
                "candidatas_heading": [],
                "indice_seccion": chunk.get("indice_seccion", 0),
                "etiqueta": lab,
                "etiqueta_secundaria": chunk.get("etiqueta"),
                "confianza": "media" if best_density >= 0.60 else "baja",
                "fuente_etiqueta": "rescate_ventana",
                "patrones_activados": [],
                "pct_rescate": round(best_density, 4),
                "patrones_rescate": sorted(patterns_found),
                "documento_id": chunk.get("documento_id", ""),
                "chunk_id": "",
            }
            nuevos.append(nuevo)

            if lab == "CONTR":
                target_contr -= 1
            else:
                target_lim -= 1

    return nuevos


def etiquetar_todos(chunks):
    """Aplica etiquetado base + rescate nivel 1 a todos los chunks."""
    for chunk in chunks:
        etiquetar_chunk(chunk)
    rescate_nivel1(chunks)
    return chunks


# ============================================================================
# MÓDULO 6: GENERACIÓN DE DATASETS
# ============================================================================


def seleccionar_para_dataset(chunks, etiqueta, target):
    """Selecciona chunks para un dataset, priorizando confianza y diversidad."""
    candidatos = [c for c in chunks if c["etiqueta"] == etiqueta
                  and MIN_CHUNK_WORDS <= c["num_palabras"] <= HARD_MAX_WORDS]

    if not candidatos:
        return []

    # Ordenar por confianza (alta > media > baja > ninguna)
    orden_confianza = {"alta": 0, "media": 1, "baja": 2, "ninguna": 3}
    candidatos.sort(key=lambda c: (orden_confianza.get(c.get("confianza", "ninguna"), 3), -c["num_palabras"]))

    # Limitar por diversidad de documentos
    doc_counts = Counter()
    seleccionados = []

    for chunk in candidatos:
        doc_id = chunk.get("documento_id", "")
        if doc_counts[doc_id] >= MAX_CHUNKS_PER_DOC_PER_LABEL:
            continue
        seleccionados.append(chunk)
        doc_counts[doc_id] += 1
        if len(seleccionados) >= target:
            break

    return seleccionados


def guardar_jsonl(chunks, ruta, etiqueta):
    """Guarda lista de chunks como JSONL."""
    with open(ruta, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            registro = {
                "chunk_id": chunk.get("chunk_id", f"{etiqueta}_{i:05d}"),
                "documento_id": chunk.get("documento_id", ""),
                "texto": chunk["texto"],
                "num_palabras": chunk["num_palabras"],
                "posicion_relativa": chunk.get("posicion_relativa", 0),
                "etiqueta": chunk.get("etiqueta", etiqueta),
                "confianza": chunk.get("confianza", ""),
                "fuente_etiqueta": chunk.get("fuente_etiqueta", ""),
                "encabezado_seccion": chunk.get("encabezado_seccion", ""),
            }
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def generar_reporte(todos_chunks, stats):
    """Genera reporte JSON con estadísticas."""
    etiquetas_count = Counter(c["etiqueta"] for c in todos_chunks)
    confianza_count = Counter(c.get("confianza", "ninguna") for c in todos_chunks)
    fuente_count = Counter(c.get("fuente_etiqueta", "") for c in todos_chunks)

    cobertura = {}
    for lab in TARGET_LABELS:
        n = etiquetas_count.get(lab, 0)
        cobertura[lab] = {
            "total_chunks": n,
            "meta_2000": n >= TARGET_PER_LABEL,
            "porcentaje_meta": round((n / TARGET_PER_LABEL) * 100, 1),
        }

    return {
        "total_chunks": len(todos_chunks),
        "total_documentos": stats.get("docs_procesados", 0),
        "docs_aceptados": stats.get("docs_aceptados", 0),
        "docs_descartados_filtro": stats.get("docs_descartados_filtro", 0),
        "docs_descartados_estructura": stats.get("docs_descartados_estructura", 0),
        "duplicados_removidos": stats.get("duplicados", 0),
        "chunks_por_etiqueta": dict(etiquetas_count.most_common()),
        "chunks_por_confianza": dict(confianza_count.most_common()),
        "chunks_por_fuente": dict(fuente_count.most_common()),
        "cobertura_meta": cobertura,
    }


def imprimir_reporte(todos_chunks, reporte):
    """Imprime reporte en consola."""
    etiquetas_count = Counter(c["etiqueta"] for c in todos_chunks)

    print("\n" + "=" * 70)
    print("REPORTE FINAL")
    print("=" * 70)
    print(f"  Documentos procesados: {reporte['total_documentos']}")
    print(f"  Documentos aceptados: {reporte['docs_aceptados']}")
    print(f"  Descartados (filtro): {reporte['docs_descartados_filtro']}")
    print(f"  Descartados (estructura): {reporte['docs_descartados_estructura']}")
    print(f"  Duplicados removidos: {reporte['duplicados_removidos']}")
    print(f"  Chunks totales: {len(todos_chunks)}")

    print("\n+-- CHUNKS POR ETIQUETA -------------------------+")
    for lab in TARGET_LABELS + ["sin_asignacion"]:
        n = etiquetas_count.get(lab, 0)
        if lab != "sin_asignacion":
            meta = "[OK]" if n >= TARGET_PER_LABEL else f"({round(n/20)}%)"
        else:
            meta = ""
        barra = "#" * min(int(n / 50), 40)
        print(f"|  {lab:<16} {n:>5}  {meta:>8}  {barra}")
    print("+------------------------------------------------+")

    print("\n+-- PROGRESO HACIA META DE 2000 -----------------+")
    for lab in TARGET_LABELS:
        info = reporte["cobertura_meta"][lab]
        n = info["total_chunks"]
        pct = info["porcentaje_meta"]
        barra = "#" * min(int(pct / 2.5), 40)
        estado = "[OK]" if info["meta_2000"] else f"{pct}%"
        print(f"|  {lab:<8} {n:>5}/2000  {estado:>8}  {barra}")
    print("+------------------------------------------------+")


# ============================================================================
# MAIN
# ============================================================================


def procesar_un_documento(ruta, texto):
    """Procesa un documento completo: encabezados → zonificación → chunks → etiquetado."""
    nombre = os.path.basename(ruta)

    # Fase 2: Identificar artículo
    encabezados = detectar_encabezados(texto)
    if not es_articulo_valido(texto, encabezados):
        return None, "no_articulo"

    # Fase 3: Zonificación
    cuerpo_limpio, inicio, fin, total_lineas = zonificar_documento(texto, encabezados)
    if not cuerpo_limpio or contar_palabras("\n".join(cuerpo_limpio)) < 200:
        return None, "cuerpo_vacio"

    # Fase 4: Segmentación
    chunks = procesar_secciones(cuerpo_limpio, total_lineas)
    if not chunks:
        return None, "sin_chunks"

    # Asignar documento_id y chunk_id
    nombre_base = nombre.replace(".txt", "")
    for i, chunk in enumerate(chunks):
        chunk["documento_id"] = nombre
        chunk["chunk_id"] = f"doc_{nombre_base}_chk{i+1:03d}"

    # Fase 5: Etiquetado
    etiquetar_todos(chunks)

    return chunks, None


def main():
    parser = argparse.ArgumentParser(description="Pipeline de Extraccion de Secciones Retoricas")
    parser.add_argument("--core_dir", type=str, default=CORE_DIR, help="Carpeta con archivos .txt")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Carpeta de salida")
    parser.add_argument("--max_docs", type=int, default=0, help="Maximo docs a procesar (0=todos)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Semilla aleatoria")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("PIPELINE DE EXTRACCION DE SECCIONES RETORICAS (STREAMING)")
    print("=" * 70)
    print(f"  Carpeta core: {args.core_dir}")
    print(f"  Salida: {args.output_dir}")

    # Inicializar contadores y limpiar archivos
    chunks_guardados = {lab: 0 for lab in TARGET_LABELS}
    chunks_por_doc = {lab: Counter() for lab in TARGET_LABELS}
    
    log.info("Inicializando archivos JSONL de salida...")
    for lab in TARGET_LABELS:
        ruta_out = os.path.join(args.output_dir, f"dataset_{lab.lower()}.jsonl")
        open(ruta_out, "w", encoding="utf-8").close()  # Limpiar archivo

    # -- Fase 1: Escanear rutas (solo rutas, no textos) -------------------
    log.info("Fase 1: Escaneando archivos...")
    archivos = [os.path.join(args.core_dir, f)
                for f in os.listdir(args.core_dir)
                if f.endswith(".txt")]
    random.shuffle(archivos)
    log.info(f"  Archivos .txt encontrados: {len(archivos)}")

    if args.max_docs > 0:
        archivos = archivos[:args.max_docs]
        log.info(f"  Limitado a: {args.max_docs}")

    # -- Fases 1-5: Procesar documento por documento -----------------------
    log.info("Fases 2-6: Procesando y escribiendo en streaming...")
    MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB max por archivo

    hashes_vistos = set()
    descartados_filtro = 0
    descartados_estructura = 0
    duplicados = 0
    errores_lectura = 0
    docs_aceptados = 0
    docs_procesados = 0
    motivos_descarte = Counter()

    for i, ruta in enumerate(archivos):
        # Progress log
        if (i + 1) % 500 == 0 or i == 0:
            guardados_str = ", ".join(f"{lab}:{chunks_guardados[lab]}" for lab in TARGET_LABELS)
            log.info(f"  {i+1}/{len(archivos)} docs | aceptados: {docs_aceptados} | {guardados_str}")

        # Saltar archivos enormes sin leerlos
        try:
            size = os.path.getsize(ruta)
            if size > MAX_FILE_BYTES:
                descartados_filtro += 1
                continue
        except OSError:
            errores_lectura += 1
            continue

        # Leer
        try:
            texto = leer_documento(ruta)
        except Exception:
            errores_lectura += 1
            continue

        docs_procesados += 1

        # Filtro rapido
        ok, motivo = filtro_rapido(texto)
        if not ok:
            descartados_filtro += 1
            del texto
            continue

        # Deduplicacion por hash
        h = hashlib.md5(" ".join(texto.split()[:500]).encode()).hexdigest()
        if h in hashes_vistos:
            duplicados += 1
            del texto
            continue
        hashes_vistos.add(h)

        # Procesar
        chunks, error = procesar_un_documento(ruta, texto)
        del texto

        if error:
            descartados_estructura += 1
            motivos_descarte[error] += 1
            continue

        docs_aceptados += 1

        # Rescate Nivel 2 en tiempo real para el documento
        if chunks_guardados["CONTR"] < TARGET_PER_LABEL or chunks_guardados["LIM"] < TARGET_PER_LABEL:
            falta_contr = TARGET_PER_LABEL - chunks_guardados["CONTR"]
            falta_lim = TARGET_PER_LABEL - chunks_guardados["LIM"]
            nuevos = rescate_nivel2(chunks, falta_contr, falta_lim)
            if nuevos:
                nombre_base = os.path.basename(ruta).replace(".txt", "")
                for j, nuevo in enumerate(nuevos):
                    nuevo["chunk_id"] = f"doc_{nombre_base}_rescue{j+1:03d}"
                    nuevo["documento_id"] = os.path.basename(ruta)
                chunks.extend(nuevos)

        # Escritura on-the-fly
        for chunk in chunks:
            lab = chunk.get("etiqueta")
            if lab in TARGET_LABELS and chunks_guardados[lab] < TARGET_PER_LABEL:
                doc_id = chunk["documento_id"]
                
                # Check constraints (min/max words + max per doc)
                if chunks_por_doc[lab][doc_id] < MAX_CHUNKS_PER_DOC_PER_LABEL:
                    if MIN_CHUNK_WORDS <= chunk["num_palabras"] <= HARD_MAX_WORDS:
                        ruta_out = os.path.join(args.output_dir, f"dataset_{lab.lower()}.jsonl")
                        with open(ruta_out, "a", encoding="utf-8") as f:
                            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        
                        chunks_guardados[lab] += 1
                        chunks_por_doc[lab][doc_id] += 1

        # Early Stopping
        if all(count >= TARGET_PER_LABEL for count in chunks_guardados.values()):
            log.info("\n" + "*"*70)
            log.info("!!! METAS ALCANZADAS !!! Todos los datasets tienen 2500 registros.")
            log.info("*"*70 + "\n")
            break

    # Reporte Final
    print("\n" + "=" * 70)
    print("REPORTE FINAL")
    print("=" * 70)
    print(f"  Documentos procesados: {docs_procesados}")
    print(f"  Documentos aceptados: {docs_aceptados}")
    print(f"  Descartados (filtro): {descartados_filtro}")
    print(f"  Descartados (estructura): {descartados_estructura}")
    print(f"  Duplicados removidos: {duplicados}")
    print(f"  Errores lectura: {errores_lectura}")
    print(f"  Motivos descarte: {dict(motivos_descarte.most_common())}")

    print("\n+-- CHUNKS GENERADOS POR ETIQUETA ---------------+ ")
    for lab in TARGET_LABELS:
        n = chunks_guardados[lab]
        meta = "[OK]" if n >= TARGET_PER_LABEL else f"({round(n/TARGET_PER_LABEL*100)}%)"
        barra = "#" * min(int((n/TARGET_PER_LABEL) * 40), 40)
        print(f"|  {lab:<16} {n:>5}  {meta:>8}  {barra}")
    print("+------------------------------------------------+")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
