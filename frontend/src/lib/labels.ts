// Metadatos de las 8 etiquetas IMRaD T1 + 2 etiquetas T2

import type { EtiquetaT1, EtiquetaT2 } from "../types/api";

export interface T1Meta {
  code: EtiquetaT1;
  name: string;
  description: string;
  hex: string;       // color exacto para chart
  borderClass: string;
  bgClass: string;
  textClass: string;
  example: string;
}

export const T1_META: Record<EtiquetaT1, T1Meta> = {
  INTRO: {
    code: "INTRO", name: "Introducción", hex: "#3B82F6",
    description: "Presenta el problema, motivación y objetivos del trabajo.",
    borderClass: "border-intro-500", bgClass: "bg-intro-50", textClass: "text-intro-700",
    example: "El objetivo de este estudio es evaluar el efecto del ejercicio aeróbico sobre la calidad del sueño en adultos mayores con insomnio crónico.",
  },
  BACK: {
    code: "BACK", name: "Antecedentes", hex: "#06B6D4",
    description: "Describe el estado del arte, marco teórico y trabajos previos.",
    borderClass: "border-back-500", bgClass: "bg-back-50", textClass: "text-back-700",
    example: "Estudios previos han demostrado que la inflamación sistémica es un factor de riesgo cardiovascular significativo en pacientes diabéticos.",
  },
  METH: {
    code: "METH", name: "Metodología", hex: "#10B981",
    description: "Diseño experimental, muestra, materiales y procedimientos.",
    borderClass: "border-meth-500", bgClass: "bg-meth-50", textClass: "text-meth-700",
    example: "Se reclutaron 200 participantes mediante muestreo aleatorio estratificado y se aplicó un cuestionario validado de 25 ítems.",
  },
  RES: {
    code: "RES", name: "Resultados", hex: "#F59E0B",
    description: "Hallazgos cuantitativos o cualitativos sin interpretación.",
    borderClass: "border-res-500", bgClass: "bg-res-50", textClass: "text-res-700",
    example: "Los resultados mostraron una reducción de 23.4% (p<0.001, IC 95%: 18.2-28.6) en los niveles de glucosa en sangre.",
  },
  DISC: {
    code: "DISC", name: "Discusión", hex: "#EF4444",
    description: "Interpretación de resultados y comparación con literatura.",
    borderClass: "border-disc-500", bgClass: "bg-disc-50", textClass: "text-disc-700",
    example: "Estos hallazgos sugieren que el efecto observado podría explicarse por la activación de vías inflamatorias previamente reportadas.",
  },
  CONC: {
    code: "CONC", name: "Conclusiones", hex: "#8B5CF6",
    description: "Síntesis final y proyección de trabajo futuro.",
    borderClass: "border-conc-500", bgClass: "bg-conc-50", textClass: "text-conc-700",
    example: "En conclusión, el tratamiento propuesto resulta efectivo y se recomienda su validación en estudios multicéntricos futuros.",
  },
  CONTR: {
    code: "CONTR", name: "Contribuciones", hex: "#EC4899",
    description: "Aportes originales explícitos del trabajo.",
    borderClass: "border-contr-500", bgClass: "bg-contr-50", textClass: "text-contr-700",
    example: "Este trabajo propone un nuevo método de detección de fraude basado en redes neuronales recurrentes con atención.",
  },
  LIM: {
    code: "LIM", name: "Limitaciones", hex: "#6B7280",
    description: "Restricciones del estudio y supuestos adoptados.",
    borderClass: "border-lim-500", bgClass: "bg-lim-50", textClass: "text-lim-700",
    example: "Sin embargo, los resultados están limitados por el tamaño reducido de la muestra y la naturaleza transversal de los datos.",
  },
};

export const T2_META: Record<EtiquetaT2, { label: string; bgClass: string; textClass: string }> = {
  contribucion: {
    label: "★ Contribución",
    bgClass: "bg-contr-500",
    textClass: "text-white",
  },
  no_contribucion: {
    label: "No-contribución",
    bgClass: "bg-gray-200",
    textClass: "text-gray-700",
  },
};

// Ejemplo multipárrafo pre-cargado para demo
export const SAMPLE_DOCUMENT = `El objetivo de este estudio es evaluar el efecto del ejercicio aeróbico sobre la calidad del sueño en adultos mayores con insomnio crónico.

Estudios previos han demostrado que la inflamación sistémica es un factor de riesgo cardiovascular significativo en pacientes con apnea del sueño.

Se reclutaron 200 participantes mediante muestreo aleatorio estratificado y se aplicó un cuestionario validado de 25 ítems durante 12 semanas.

Los resultados mostraron una reducción de 23.4% (p<0.001, IC 95%: 18.2-28.6) en los puntajes del Índice de Calidad del Sueño de Pittsburgh.

Este trabajo propone un nuevo método de evaluación del sueño basado en sensores no invasivos que combinan análisis de movimiento y patrón respiratorio.

Sin embargo, los resultados están limitados por el tamaño reducido de la muestra y la naturaleza transversal de los datos, que impiden establecer relaciones causales.`;
