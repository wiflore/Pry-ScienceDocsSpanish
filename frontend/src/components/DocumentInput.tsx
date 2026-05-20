import { useState } from "react";
import { Loader2, Sparkles, FileText } from "lucide-react";
import type { ModeloFamilia } from "../types/api";
import ModelFamilySelector from "./ModelFamilySelector";
import { SAMPLE_DOCUMENT } from "../lib/labels";

interface Props {
  onSubmit: (texto: string, modelo: ModeloFamilia) => void;
  loading: boolean;
  geminiEnabled: boolean;
  initialText?: string;
  initialModel?: ModeloFamilia;
}

export default function DocumentInput({
  onSubmit,
  loading,
  geminiEnabled,
  initialText = "",
  initialModel = "scibeto",
}: Props) {
  const [texto, setTexto] = useState(initialText);
  const [modelo, setModelo] = useState<ModeloFamilia>(initialModel);

  const numParrafos = texto
    .split(/\n\n+/)
    .filter((p) => p.trim().length > 0).length;

  return (
    <div className="space-y-4 bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-2">
          Texto del artículo científico
        </label>
        <p className="text-xs text-gray-500 mb-3">
          Pega texto completo o parcial en español. Separa los párrafos con
          línea en blanco para mejor segmentación.
        </p>
        <textarea
          value={texto}
          onChange={(e) => setTexto(e.target.value)}
          rows={12}
          className="w-full rounded-lg border border-gray-300 p-3 font-serif text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-y"
          placeholder="Ejemplo:&#10;&#10;El objetivo de este estudio es evaluar el efecto del ejercicio aeróbico...&#10;&#10;Este trabajo propone un nuevo método de evaluación del sueño...&#10;&#10;Sin embargo, los resultados están limitados por el tamaño de la muestra..."
        />
        <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
          <span>
            {texto.length} caracteres ·{" "}
            <strong>{numParrafos}</strong> párrafo
            {numParrafos !== 1 ? "s" : ""}
          </span>
          <button
            type="button"
            onClick={() => setTexto(SAMPLE_DOCUMENT)}
            className="flex items-center gap-1.5 text-blue-600 hover:text-blue-700 font-medium"
          >
            <FileText className="h-3.5 w-3.5" />
            Cargar ejemplo
          </button>
        </div>
      </div>

      <div>
        <label className="block text-sm font-semibold text-gray-900 mb-2">
          Familia de modelo a usar
        </label>
        <ModelFamilySelector
          value={modelo}
          onChange={setModelo}
          geminiEnabled={geminiEnabled}
        />
      </div>

      <button
        type="button"
        onClick={() => onSubmit(texto, modelo)}
        disabled={loading || !texto.trim()}
        className="w-full flex items-center justify-center gap-2 rounded-lg bg-blue-600 hover:bg-blue-700 text-white py-3 font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            {modelo === "gemini" ? "Analizando con Gemini (~10s)..." : "Analizando..."}
          </>
        ) : (
          <>
            <Sparkles className="h-5 w-5" />
            Analizar T1 + T2
          </>
        )}
      </button>
    </div>
  );
}
