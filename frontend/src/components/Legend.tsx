import { T1_META } from "../lib/labels";

export default function Legend() {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <div className="text-xs font-semibold text-gray-600 mb-2">
        Etiquetas T1 (IMRaD)
      </div>
      <div className="flex flex-wrap gap-2">
        {Object.values(T1_META).map((m) => (
          <span
            key={m.code}
            className={`inline-flex items-center gap-1.5 px-2 py-1 rounded text-xs font-medium ${m.bgClass} ${m.textClass} border ${m.borderClass}`}
            title={m.description}
          >
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: m.hex }}
            />
            {m.code} — {m.name}
          </span>
        ))}
      </div>
    </div>
  );
}
