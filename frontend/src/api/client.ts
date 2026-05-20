// Cliente HTTP minimalista basado en fetch.
// Usa el proxy de Vite en dev (/api → backend AWS) para evitar CORS.
// En prod, VITE_API_PATH queda vacío y se llama directo al BASE.

import type {
  PredictRequest,
  PredictionResponse,
  AnalisisRequest,
  AnalisisResponse,
  HealthResponse,
  ModelosResponse,
} from "../types/api";

const PATH_PREFIX = import.meta.env.VITE_API_PATH ?? "";
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

// En dev: PATH_PREFIX = "/api" → fetch("/api/health") → proxy de Vite reescribe a backend
// En prod: PATH_PREFIX = "" → fetch("http://backend/health") directo (requiere CORS)
const BASE = PATH_PREFIX || BASE_URL;

async function postJSON<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
  return res.json() as Promise<TRes>;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

// ─── Endpoints ──────────────────────────────────────────────────────────────
export const health = () => getJSON<HealthResponse>("/health");
export const modelos = () => getJSON<ModelosResponse>("/modelos");
export const clasificar = (req: PredictRequest) =>
  postJSON<PredictRequest, PredictionResponse>("/clasificar", req);
export const contribucion = (req: PredictRequest) =>
  postJSON<PredictRequest, PredictionResponse>("/contribucion", req);
export const analizar = (req: AnalisisRequest) =>
  postJSON<AnalisisRequest, AnalisisResponse>("/analizar", req);
