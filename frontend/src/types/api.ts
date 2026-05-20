// Tipos del contrato API v2.0 — Pry-ScienceDocsSpanish
// Validados contra http://100.27.211.193:8000

export type EtiquetaT1 =
  | "INTRO"
  | "BACK"
  | "METH"
  | "RES"
  | "DISC"
  | "CONC"
  | "CONTR"
  | "LIM";

export type EtiquetaT2 = "contribucion" | "no_contribucion";

// Decisión del equipo: solo 2 familias en producción (Qwen es TODO)
export type ModeloFamilia = "scibeto" | "gemini";

// ─── /clasificar (T1) y /contribucion (T2) ──────────────────────────────────
export interface PredictRequest {
  texto: string;
  modelo: ModeloFamilia;
}

export interface PredictionResponse {
  etiqueta: string;
  confianza: number;
  probabilidades: Record<string, number>;
  modelo_usado: string;
}

// ─── /analizar (pipeline T1+T2 sobre texto largo) ───────────────────────────
export interface AnalisisRequest {
  texto: string;
  modelo: ModeloFamilia;
}

export interface FragmentoResult {
  fragmento: string;
  t1: EtiquetaT1;
  confianza_t1: number;
  probabilidades_t1: Record<EtiquetaT1, number>;
  t2: EtiquetaT2;
  confianza_t2: number;
  probabilidades_t2: Record<EtiquetaT2, number>;
  modelo_t1: string;
  modelo_t2: string;
}

export interface AnalisisResponse {
  n_fragmentos: number;
  modelo: ModeloFamilia;
  fragmentos: FragmentoResult[];
}

// ─── /health ────────────────────────────────────────────────────────────────
export interface HealthResponse {
  status: "ok";
  t1_model: string;
  t2_model: string;
  gemini_key_set: boolean;
}

// ─── /modelos ───────────────────────────────────────────────────────────────
export interface ModelosResponse {
  t1: { endpoint: string; modelos: ModeloFamilia[] };
  t2: { endpoint: string; modelos: ModeloFamilia[] };
  pipeline: { endpoint: string; modelos: ModeloFamilia[] };
}
