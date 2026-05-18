"""
Script para calcular el Cohen's Kappa entre las columnas
'etiqueta_heuristica' y 'etiqueta_manual' del archivo
tarea2_muestra_anotacion_manual.csv
"""

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import os

# --- Configuracion ---
ARCHIVO = os.path.join(os.path.dirname(__file__), "tarea2_muestra_anotacion_manual.csv")
COL_ORIGINAL = "etiqueta_heuristica"
COL_ANOTADOR = "etiqueta_manual"

# --- Lectura del archivo ---
print(f"Leyendo archivo: {ARCHIVO}\n")
df = pd.read_csv(ARCHIVO, sep=";", encoding="utf-8-sig")

# Verificar que las columnas existan
for col in [COL_ORIGINAL, COL_ANOTADOR]:
    if col not in df.columns:
        raise ValueError(f"La columna '{col}' no se encontro. Columnas disponibles: {list(df.columns)}")

# Eliminar filas con valores nulos en las columnas de interes
n_total = len(df)
df_clean = df.dropna(subset=[COL_ORIGINAL, COL_ANOTADOR]).copy()
n_nulos = n_total - len(df_clean)
if n_nulos > 0:
    print(f"[AVISO] Se eliminaron {n_nulos} filas con valores nulos.\n")

# Normalizar etiquetas (minusculas y sin espacios extra)
df_clean[COL_ORIGINAL] = df_clean[COL_ORIGINAL].astype(str).str.strip().str.lower()
df_clean[COL_ANOTADOR] = df_clean[COL_ANOTADOR].astype(str).str.strip().str.lower()

y_original = df_clean[COL_ORIGINAL]
y_anotador = df_clean[COL_ANOTADOR]

# --- Cohen's Kappa ---
kappa = cohen_kappa_score(y_original, y_anotador)

print("=" * 60)
print("           RESULTADOS - COHEN'S KAPPA (TAREA 2)")
print("=" * 60)
print(f"\n  Total de muestras evaluadas: {len(df_clean)}")
print(f"  Cohen's Kappa:               {kappa:.4f}")

# Interpretacion del valor
if kappa < 0:
    interpretacion = "Sin acuerdo (peor que el azar)"
elif kappa < 0.20:
    interpretacion = "Acuerdo insignificante"
elif kappa < 0.40:
    interpretacion = "Acuerdo bajo"
elif kappa < 0.60:
    interpretacion = "Acuerdo moderado"
elif kappa < 0.80:
    interpretacion = "Acuerdo sustancial"
else:
    interpretacion = "Acuerdo casi perfecto"

print(f"  Interpretacion:              {interpretacion}")

# --- Porcentaje de acuerdo simple ---
acuerdo = (y_original == y_anotador).sum()
porcentaje_acuerdo = acuerdo / len(df_clean) * 100
print(f"\n  Acuerdo simple:              {acuerdo}/{len(df_clean)} ({porcentaje_acuerdo:.2f}%)")

# --- Etiquetas unicas ---
etiquetas = sorted(set(y_original) | set(y_anotador))
print(f"\n  Etiquetas unicas:            {len(etiquetas)}")
for et in etiquetas:
    print(f"    - {et}")

# --- Matriz de confusion ---
print("\n" + "=" * 60)
print("           MATRIZ DE CONFUSION")
print("=" * 60)
cm = confusion_matrix(y_original, y_anotador, labels=etiquetas)
cm_df = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)
print(f"\n(Filas = {COL_ORIGINAL}, Columnas = {COL_ANOTADOR})\n")
print(cm_df.to_string())

# --- Reporte de clasificacion ---
print("\n" + "=" * 60)
print("           REPORTE DE CLASIFICACION")
print("=" * 60)
print(f"\n(Tomando '{COL_ORIGINAL}' como referencia)\n")
print(classification_report(y_original, y_anotador, labels=etiquetas, zero_division=0))

# --- Kappa por categoria (one-vs-rest) ---
print("=" * 60)
print("           KAPPA POR CATEGORIA (one-vs-rest)")
print("=" * 60)
for et in etiquetas:
    y_orig_bin = (y_original == et).astype(int)
    y_anot_bin = (y_anotador == et).astype(int)
    k = cohen_kappa_score(y_orig_bin, y_anot_bin)
    print(f"  {et:40s}  k = {k:.4f}")

print("\n" + "=" * 60)
print("  Calculo completado exitosamente.")
print("=" * 60)
