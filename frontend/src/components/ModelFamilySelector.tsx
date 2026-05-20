import type { ModeloFamilia } from "../types/api";
import { Cpu, Cloud } from "lucide-react";

interface Props {
  value: ModeloFamilia;
  onChange: (m: ModeloFamilia) => void;
  geminiEnabled: boolean;
}

export default function ModelFamilySelector({
  value,
  onChange,
  geminiEnabled,
}: Props) {
  return (
    <div className="grid grid-cols-2 gap-3">
      <button
        type="button"
        onClick={() => onChange("scibeto")}
        className={`flex items-center gap-3 px-4 py-3 rounded-lg border-2 transition-all text-left ${
          value === "scibeto"
            ? "border-blue-600 bg-blue-50 ring-2 ring-blue-200"
            : "border-gray-200 bg-white hover:border-gray-300"
        }`}
      >
        <Cpu
          className={`h-6 w-6 ${
            value === "scibeto" ? "text-blue-600" : "text-gray-400"
          }`}
        />
        <div>
          <div className="font-semibold text-sm">Encoder</div>
          <div className="text-xs text-gray-500">SciBETO · rápido · local</div>
        </div>
      </button>

      <button
        type="button"
        onClick={() => geminiEnabled && onChange("gemini")}
        disabled={!geminiEnabled}
        title={
          !geminiEnabled
            ? "Gemini no disponible: GEMINI_API_KEY no configurada en el backend"
            : "Modelo comercial vía API"
        }
        className={`flex items-center gap-3 px-4 py-3 rounded-lg border-2 transition-all text-left ${
          !geminiEnabled
            ? "border-gray-200 bg-gray-50 opacity-50 cursor-not-allowed"
            : value === "gemini"
            ? "border-purple-600 bg-purple-50 ring-2 ring-purple-200"
            : "border-gray-200 bg-white hover:border-gray-300"
        }`}
      >
        <Cloud
          className={`h-6 w-6 ${
            value === "gemini" ? "text-purple-600" : "text-gray-400"
          }`}
        />
        <div>
          <div className="font-semibold text-sm">Comercial</div>
          <div className="text-xs text-gray-500">Gemini 2.5 Flash · API</div>
        </div>
      </button>
    </div>
  );
}
