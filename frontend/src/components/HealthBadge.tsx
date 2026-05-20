import { useEffect, useState } from "react";
import { CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { health } from "../api/client";
import type { HealthResponse } from "../types/api";

interface Props {
  onHealth?: (h: HealthResponse) => void;
}

export default function HealthBadge({ onHealth }: Props) {
  const [state, setState] = useState<"loading" | "ok" | "down">("loading");
  const [info, setInfo] = useState<HealthResponse | null>(null);

  useEffect(() => {
    health()
      .then((h) => {
        setState("ok");
        setInfo(h);
        onHealth?.(h);
      })
      .catch(() => setState("down"));
  }, []);

  if (state === "loading")
    return (
      <span className="flex items-center gap-1.5 text-sm text-gray-500">
        <Loader2 className="h-4 w-4 animate-spin" /> verificando…
      </span>
    );
  if (state === "down")
    return (
      <span className="flex items-center gap-1.5 text-sm text-red-600 font-medium">
        <XCircle className="h-4 w-4" /> API no disponible
      </span>
    );
  return (
    <span className="flex items-center gap-1.5 text-sm text-green-700 font-medium">
      <CheckCircle2 className="h-4 w-4" />
      API en línea
      <span className="text-xs text-gray-500 font-normal">
        · Gemini {info?.gemini_key_set ? "✓" : "✗"}
      </span>
    </span>
  );
}
