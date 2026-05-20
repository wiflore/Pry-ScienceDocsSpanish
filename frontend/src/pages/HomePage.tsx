import { Link } from "react-router-dom";
import {
  Sparkles,
  Cpu,
  ArrowRight,
  FileText,
  Star,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="space-y-12">
      {/* Hero */}
      <section className="text-center space-y-4 py-8">
        <h1 className="text-4xl md:text-5xl font-bold text-gray-900 leading-tight">
          Analizador retórico para
          <br />
          <span className="text-blue-600">artículos científicos en español</span>
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          Clasifica cada párrafo en su categoría IMRaD y detecta automáticamente
          si declara una contribución científica explícita. Dos familias de
          modelos al alcance de un clic.
        </p>
        <div className="flex items-center justify-center gap-3 pt-4">
          <Link
            to="/analizar"
            className="inline-flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-semibold transition-colors"
          >
            <Sparkles className="h-5 w-5" />
            Probar el analizador
          </Link>
          <Link
            to="/arquitectura"
            className="inline-flex items-center gap-2 text-gray-700 hover:text-gray-900 px-6 py-3 rounded-lg font-medium"
          >
            Ver arquitectura
            <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
      </section>

      {/* 3 cards */}
      <section className="grid md:grid-cols-3 gap-4">
        <FeatureCard
          icon={<FileText className="h-6 w-6" />}
          title="Tarea 1 — Segmentación retórica"
          desc="Cada párrafo recibe una de 8 etiquetas IMRaD: introducción, antecedentes, metodología, resultados, discusión, conclusiones, contribuciones, limitaciones."
        />
        <FeatureCard
          icon={<Star className="h-6 w-6" />}
          title="Tarea 2 — Detección de contribuciones"
          desc="Sobre el mismo texto, un clasificador binario decide si cada párrafo declara explícitamente un aporte propio del trabajo."
        />
        <FeatureCard
          icon={<Cpu className="h-6 w-6" />}
          title="Dos familias de modelos"
          desc="Encoder fine-tuneado (SciBETO en español científico) para baja latencia, o modelo comercial (Gemini 2.5 Flash) vía API. Cambias con un clic."
        />
      </section>

      {/* Cómo funciona */}
      <section className="bg-white rounded-2xl border border-gray-200 p-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-6 text-center">
          Cómo funciona
        </h2>
        <div className="grid md:grid-cols-4 gap-6">
          <Step n={1} title="Pegas tu texto" desc="Párrafos separados por línea en blanco" />
          <Step n={2} title="Eliges modelo" desc="Encoder local o comercial vía API" />
          <Step n={3} title="Pipeline T1+T2" desc="El backend segmenta y clasifica párrafo por párrafo" />
          <Step n={4} title="Resultados visuales" desc="Cada párrafo con su color y badge de contribución" />
        </div>
      </section>

      {/* Tech stack */}
      <section className="text-center pb-8">
        <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">
          Construido con
        </h3>
        <div className="flex items-center justify-center gap-4 flex-wrap text-sm text-gray-600">
          <span>SciBETO-large</span> · <span>FastAPI</span> ·
          <span>AWS ECS Fargate</span> · <span>HuggingFace Hub</span> ·
          <span>React + Vite</span> · <span>Gemini 2.5 Flash</span>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  desc,
}: {
  icon: React.ReactNode;
  title: string;
  desc: string;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-6 hover:shadow-md transition-shadow">
      <div className="w-12 h-12 bg-blue-50 text-blue-600 rounded-lg flex items-center justify-center mb-3">
        {icon}
      </div>
      <h3 className="font-semibold text-gray-900 mb-2">{title}</h3>
      <p className="text-sm text-gray-600 leading-relaxed">{desc}</p>
    </div>
  );
}

function Step({ n, title, desc }: { n: number; title: string; desc: string }) {
  return (
    <div className="flex flex-col items-center text-center">
      <div className="w-10 h-10 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold mb-3">
        {n}
      </div>
      <div className="font-semibold text-gray-900 text-sm mb-1">{title}</div>
      <div className="text-xs text-gray-600">{desc}</div>
    </div>
  );
}
