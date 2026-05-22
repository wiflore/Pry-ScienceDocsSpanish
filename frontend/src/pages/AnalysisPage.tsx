import { useState } from "react";
import { AlertCircle, Clock, Cpu } from "lucide-react";
import { analizar, ApiError } from "../api/client";
import type {
  AnalisisResponse,
  HealthResponse,
  ModeloFamilia,
} from "../types/api";
import DocumentInput from "../components/DocumentInput";
import ParagraphCard from "../components/ParagraphCard";
import HealthBadge from "../components/HealthBadge";
import Legend from "../components/Legend";

export default function AnalysisPage() {
  const [result, setResult] = useState<AnalisisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [geminiEnabled, setGeminiEnabled] = useState(false);

  function onHealth(h: HealthResponse) {
    setGeminiEnabled(h.gemini_key_set);
  }

  async function handleAnalyze(texto: string, modelo: ModeloFamilia) {
    setError(null);
    setResult(null);
    setLatencyMs(null);
    setLoading(true);
    const t0 = performance.now();
    try {
      const r = await analizar({ texto, modelo });
      setLatencyMs(Math.round(performance.now() - t0));
      if (r.n_fragmentos === 0) {
        setError(
          "El texto no contiene párrafos detectables. Asegúrate de separarlos con línea en blanco (presiona Enter dos veces)."
        );
      } else {
        setResult(r);
      }
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 503
            ? "El modelo seleccionado no está disponible en este momento. Prueba con el Encoder (SciBETO)."
            : `Error ${e.status}: ${e.detail.slice(0, 200)}`
        );
      } else {
        setError(
          e instanceof Error
            ? e.message
            : "No se pudo conectar al backend. Verifica tu conexión."
        );
      }
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <header className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">
            Analizador retórico + Contribuciones
          </h1>
          <p className="text-sm text-gray-600">
            Pipeline integrado T1 (segmentación IMRaD) + T2 (detección binaria
            de contribución).
          </p>
        </div>
        <HealthBadge onHealth={onHealth} />
      </header>

      <DocumentInput
        onSubmit={handleAnalyze}
        loading={loading}
        geminiEnabled={geminiEnabled}
      />

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg flex items-start gap-2">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <p className="text-sm">{error}</p>
        </div>
      )}

      {result && (
        <section className="space-y-4">
          <div className="flex items-center justify-between text-sm text-gray-600 px-1">
            <div className="flex items-center gap-4">
              <span>
                <strong className="text-gray-900">{result.n_fragmentos}</strong>{" "}
                fragmento{result.n_fragmentos !== 1 ? "s" : ""} analizados
              </span>
              <span className="inline-flex items-center gap-1.5">
                <Cpu className="h-4 w-4" />
                <code className="text-xs bg-gray-100 px-1.5 py-0.5 rounded">
                  {result.modelo}
                </code>
              </span>
              {latencyMs !== null && (
                <span className="inline-flex items-center gap-1.5">
                  <Clock className="h-4 w-4" />
                  {latencyMs} ms
                </span>
              )}
            </div>
          </div>

          <Legend />

          <div className="space-y-3">
            {result.fragmentos.map((frag, idx) => (
              <ParagraphCard key={idx} frag={frag} idx={idx} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
