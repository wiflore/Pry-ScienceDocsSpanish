import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import Header from "./components/Header";
import HomePage from "./pages/HomePage";
import AnalysisPage from "./pages/AnalysisPage";
import ArchitecturePage from "./pages/ArchitecturePage";
import AboutPage from "./pages/AboutPage";
import uniandesLogo from "./assets/uniandes-logo.png";

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <Header />
        <main className="flex-1 max-w-6xl mx-auto w-full px-4 md:px-6 py-6 md:py-10">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/analizar" element={<AnalysisPage />} />
            <Route path="/arquitectura" element={<ArchitecturePage />} />
            <Route path="/sobre" element={<AboutPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <footer className="bg-white border-t border-gray-200 py-5 mt-8">
          <div className="max-w-6xl mx-auto px-6 flex flex-col items-center gap-3">
            <img
              src={uniandesLogo}
              alt="Universidad de los Andes"
              className="h-10 w-auto"
            />
            <div className="text-center text-xs text-gray-500">
              Pry-ScienceDocsSpanish · Maestría en IA Aplicada — Universidad de
              los Andes ·{" "}
              <a
                href="https://huggingface.co/wiflore/SciBETO-IMRaD"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Modelo T1 en HF
              </a>
              {" · "}
              <a
                href="https://huggingface.co/wiflore/SciBETO-T2-contribucion"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:underline"
              >
                Modelo T2 en HF
              </a>
            </div>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}
