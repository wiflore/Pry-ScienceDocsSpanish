import { useState } from "react";
import { ChevronDown, ChevronUp, Star } from "lucide-react";
import type { FragmentoResult } from "../types/api";
import { T1_META, T2_META } from "../lib/labels";
import ProbabilityChart from "./ProbabilityChart";

interface Props {
  frag: FragmentoResult;
  idx: number;
}

export default function ParagraphCard({ frag, idx }: Props) {
  const [expanded, setExpanded] = useState(false);
  const t1 = T1_META[frag.t1];
  const t2 = T2_META[frag.t2];
  const isContrib = frag.t2 === "contribucion";

  return (
    <div
      className={`rounded-xl border-l-4 ${t1.borderClass} ${t1.bgClass} p-4 transition-shadow hover:shadow-md`}
    >
      {/* Header con badges */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-semibold text-gray-500">
            #{idx + 1}
          </span>
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold ${t1.textClass} bg-white border ${t1.borderClass}`}
            title={t1.description}
          >
            {t1.code}
            <span className="font-normal">
              · {(frag.confianza_t1 * 100).toFixed(0)}%
            </span>
          </span>
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${t2.bgClass} ${t2.textClass}`}
          >
            {isContrib && <Star className="h-3 w-3 fill-current" />}
            {t2.label}
            <span className="font-normal opacity-80">
              · {(frag.confianza_t2 * 100).toFixed(0)}%
            </span>
          </span>
        </div>
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="text-gray-400 hover:text-gray-600"
          title={expanded ? "Ocultar detalles" : "Ver probabilidades"}
        >
          {expanded ? (
            <ChevronUp className="h-5 w-5" />
          ) : (
            <ChevronDown className="h-5 w-5" />
          )}
        </button>
      </div>

      {/* Texto del fragmento */}
      <p className="text-sm leading-relaxed text-gray-800 font-serif">
        {frag.fragmento}
      </p>

      {/* Detalles expandibles */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-gray-200 space-y-3">
          <div>
            <div className="text-xs font-semibold text-gray-600 mb-1">
              {t1.name} ({t1.code}) — probabilidades T1
            </div>
            <ProbabilityChart
              probabilidades={frag.probabilidades_t1}
              type="t1"
            />
          </div>
          <div>
            <div className="text-xs font-semibold text-gray-600 mb-1">
              Contribución (T2) — probabilidades
            </div>
            <ProbabilityChart
              probabilidades={frag.probabilidades_t2}
              type="t2"
            />
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs text-gray-500 pt-2">
            <div>
              <span className="font-semibold">Modelo T1:</span>{" "}
              <code className="text-[10px]">{frag.modelo_t1}</code>
            </div>
            <div>
              <span className="font-semibold">Modelo T2:</span>{" "}
              <code className="text-[10px]">{frag.modelo_t2}</code>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
