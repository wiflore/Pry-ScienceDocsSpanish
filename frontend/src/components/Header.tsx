import { NavLink } from "react-router-dom";
import { FlaskConical } from "lucide-react";

export default function Header() {
  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? "bg-blue-600 text-white"
        : "text-gray-700 hover:bg-gray-100"
    }`;

  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
      <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
        <NavLink to="/" className="flex items-center gap-2 text-gray-900">
          <FlaskConical className="h-6 w-6 text-blue-600" />
          <span className="font-bold text-lg">SciDocs<span className="text-blue-600">Spanish</span></span>
        </NavLink>
        <nav className="flex items-center gap-1">
          <NavLink to="/" end className={linkClass}>Inicio</NavLink>
          <NavLink to="/analizar" className={linkClass}>Analizador</NavLink>
          <NavLink to="/arquitectura" className={linkClass}>Arquitectura</NavLink>
          <NavLink to="/sobre" className={linkClass}>Sobre</NavLink>
        </nav>
      </div>
    </header>
  );
}
