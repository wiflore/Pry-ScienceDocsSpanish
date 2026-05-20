import { ExternalLink } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <header>
        <h1 className="text-2xl font-bold text-gray-900">Sobre el proyecto</h1>
        <p className="text-sm text-gray-600 mt-1">
          Análisis retórico y extracción de contribuciones en artículos
          científicos en español.
        </p>
      </header>

      <section className="bg-white rounded-xl border border-gray-200 p-6 space-y-3">
        <h2 className="font-semibold text-gray-900">Contexto académico</h2>
        <p className="text-sm text-gray-700 leading-relaxed">
          Proyecto final de la Maestría en Inteligencia Artificial Aplicada de
          la Universidad de los Andes. Grupo FLAG-TICsW. Aborda dos tareas de
          NLP sobre un corpus anotado humanamente de papers científicos en
          español:
        </p>
        <ul className="text-sm text-gray-700 list-disc pl-6 space-y-1">
          <li>
            <strong>Tarea 1</strong> — clasificación retórica de fragmentos en
            8 categorías IMRaD extendidas (INTRO, BACK, METH, RES, DISC, CONC,
            CONTR, LIM).
          </li>
          <li>
            <strong>Tarea 2</strong> — detección binaria de contribución
            científica explícita.
          </li>
        </ul>
      </section>

      <section className="bg-white rounded-xl border border-gray-200 p-6 space-y-3">
        <h2 className="font-semibold text-gray-900">Modelos publicados</h2>
        <ul className="text-sm space-y-2">
          <ExternalLi
            href="https://huggingface.co/wiflore/SciBETO-IMRaD"
            label="wiflore/SciBETO-IMRaD"
            desc="T1: encoder SciBETO-large fine-tuneado para IMRaD-8"
          />
          <ExternalLi
            href="https://huggingface.co/wiflore/SciBETO-T2-contribucion"
            label="wiflore/SciBETO-T2-contribucion"
            desc="T2: encoder SciBETO-large fine-tuneado binario (F1-pos 0.84)"
          />
        </ul>
      </section>

      <section className="bg-white rounded-xl border border-gray-200 p-6 space-y-3">
        <h2 className="font-semibold text-gray-900">Equipo</h2>
        <div className="text-sm text-gray-700 space-y-1">
          <p>
            <strong>Desarrollador de modelos:</strong> Anderson — entrenamiento
            de SciBETO T1 y T2, integración Gemini en la API.
          </p>
          <p>
            <strong>Ingeniero de implementación + Frontend:</strong> Daniel
            Felipe Caro — despliegue en AWS, frontend React, documentación
            operativa.
          </p>
          <p className="text-xs text-gray-500 pt-2">
            Más integrantes en el documento de proyecto entregado a la
            universidad.
          </p>
        </div>
      </section>

      <section className="text-center text-xs text-gray-500 pb-4">
        Versión de la API: v2.0.0 · Última actualización: mayo 2026
      </section>
    </div>
  );
}

function ExternalLi({
  href,
  label,
  desc,
}: {
  href: string;
  label: string;
  desc: string;
}) {
  return (
    <li>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 hover:underline inline-flex items-center gap-1 font-medium"
      >
        {label}
        <ExternalLink className="h-3.5 w-3.5" />
      </a>
      <div className="text-xs text-gray-600 mt-0.5">{desc}</div>
    </li>
  );
}
