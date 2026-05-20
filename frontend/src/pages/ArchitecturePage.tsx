export default function ArchitecturePage() {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <header>
        <h1 className="text-2xl font-bold text-gray-900">Arquitectura técnica</h1>
        <p className="text-sm text-gray-600 mt-1">
          Cómo se conectan los componentes desde el navegador hasta el modelo.
        </p>
      </header>

      {/* Diagrama */}
      <section className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="font-semibold text-gray-900 mb-4">Diagrama de despliegue</h2>
        <pre className="bg-gray-50 border border-gray-200 rounded p-4 text-xs font-mono overflow-x-auto leading-relaxed">
{`USUARIO / EVALUADOR
        │ HTTPS
        ▼
React Frontend (AWS Amplify Hosting)   ← este sitio
        │ fetch JSON
        ▼
FastAPI v2.0  (AWS ECS Fargate, us-east-1)
   2 vCPU · 6 GB RAM · imagen :v2.0.0
        │
        ├── /clasificar    (T1)
        ├── /contribucion  (T2)
        ├── /analizar      (pipeline T1+T2 segmentando por \\n\\n)
        ├── /modelos
        └── /health
        │
        ├──► SciBETO-IMRaD            (Hugging Face Hub)
        ├──► SciBETO-T2-contribucion  (Hugging Face Hub)
        └──► Gemini 2.5 Flash         (Google AI API)`}
        </pre>
      </section>

      {/* Decisiones */}
      <section className="grid md:grid-cols-2 gap-4">
        <DecisionCard
          title="Encoder fine-tuneado para producción"
          desc="SciBETO-large pre-entrenado en texto científico en español, fine-tuneado para 8 clases IMRaD (T1, macro-F1 ~0.40) y para contribución binaria (T2, F1-pos 0.84)."
        />
        <DecisionCard
          title="Comercial como alternativa de top-line"
          desc="Gemini 2.5 Flash vía API. Mejores predicciones en T1 (Gemini few-shot ~0.49), latencia mayor (~3 s por llamada) y costo marginal por uso."
        />
        <DecisionCard
          title="Selector dinámico"
          desc={`El usuario elige la familia para cada análisis. La API expone ambos modelos en los tres endpoints POST con el campo "modelo".`}
        />
        <DecisionCard
          title="Pipeline integrado /analizar"
          desc="Segmenta el texto por línea en blanco y corre T1 + T2 sobre cada párrafo en una sola request. El response trae la lista de fragmentos con etiquetas y probabilidades."
        />
        <DecisionCard
          title="Versionamiento sin latest"
          desc="Cada deploy referencia la imagen ECR por tag semántico (v2.0.0) más tag de fecha (v2.0.0-20260517). Sin tag latest para garantizar reproducibilidad."
        />
        <DecisionCard
          title="Secretos en Parameter Store"
          desc="La GEMINI_API_KEY vive en AWS Systems Manager Parameter Store como SecureString. Se inyecta a la task ECS como variable de entorno al runtime."
        />
      </section>

      {/* Stack */}
      <section className="bg-white rounded-xl border border-gray-200 p-6">
        <h2 className="font-semibold text-gray-900 mb-3">Stack tecnológico</h2>
        <ul className="text-sm text-gray-700 space-y-1.5">
          <li><strong>Modelos:</strong> SciBETO-large, Gemini 2.5 Flash</li>
          <li><strong>Backend:</strong> FastAPI + Uvicorn (Python 3.11) sobre Docker</li>
          <li><strong>Frontend:</strong> React 18 + Vite + TypeScript + Tailwind CSS</li>
          <li><strong>Hosting modelos:</strong> Hugging Face Hub</li>
          <li><strong>Hosting API:</strong> AWS ECS Fargate (region us-east-1)</li>
          <li><strong>Hosting frontend:</strong> AWS Amplify (próximo)</li>
          <li><strong>Visualización:</strong> Recharts + lucide-react</li>
        </ul>
      </section>
    </div>
  );
}

function DecisionCard({ title, desc }: { title: string; desc: string }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-5">
      <div className="font-semibold text-gray-900 text-sm mb-1.5">{title}</div>
      <div className="text-sm text-gray-600 leading-relaxed">{desc}</div>
    </div>
  );
}
