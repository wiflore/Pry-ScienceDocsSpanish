import diagramaDespliegue from "../assets/diagrama-despliegue.png";

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
        <img
          src={diagramaDespliegue}
          alt="Diagrama de despliegue: navegador, frontend React en AWS Amplify, API FastAPI v2.0.2 en AWS ECS Fargate y modelos en Hugging Face Hub y Gemini"
          className="w-full rounded border border-gray-200"
        />
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
          desc="Cada deploy referencia la imagen ECR por tag semántico (v2.0.2) más tag de fecha (v2.0.2-20260520). Sin tag latest para garantizar reproducibilidad."
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
