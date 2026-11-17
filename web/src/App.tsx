import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { UNAUTHORIZED_EVENT, api } from "./api";
import { useAsync } from "./hooks";
import NewRun from "./pages/NewRun";
import RunDetail from "./pages/RunDetail";
import Runs from "./pages/Runs";
import System from "./pages/System";
import { SchemaContext } from "./schema";

const NAV = [
  { to: "/", label: "Runs", end: true },
  { to: "/new", label: "New run" },
  { to: "/system", label: "System" },
];

export default function App() {
  const schema = useAsync(() => api.schema(), []);
  // The server's sign-in message, once any API call was refused for a missing or outdated session.
  const [signIn, setSignIn] = useState<string | null>(null);
  useEffect(() => {
    const onUnauthorized = (e: Event) => setSignIn((e as CustomEvent<string>).detail);
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
  }, []);
  return (
    <SchemaContext.Provider value={schema.data}>
      <div className="flex h-full flex-col">
        <header className="flex items-center gap-6 border-b border-slate-800 bg-slate-950 px-5 py-2.5">
          <NavLink to="/" className="text-base font-bold tracking-tight text-white">
            mono<span className="text-sky-400">rl</span>
          </NavLink>
          <nav className="flex gap-1">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.end}
                className={({ isActive }) => `rounded-md px-3 py-1 text-sm ${isActive ? "bg-slate-800 text-white" : "text-slate-400 hover:text-slate-200"}`}
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          {schema.error && !signIn && <span className="ml-auto text-xs text-rose-400">API unreachable: {schema.error}</span>}
        </header>
        <main className="min-h-0 flex-1 overflow-auto">
          {signIn ? (
            <div className="mx-auto mt-16 max-w-xl rounded-lg border border-slate-800 bg-slate-900 p-6">
              <h1 className="text-lg font-semibold text-white">Sign in to monorl</h1>
              <p className="mt-3 text-sm leading-6 text-slate-300">{signIn}</p>
            </div>
          ) : (
            <Routes>
              <Route path="/" element={<Runs />} />
              <Route path="/new" element={<NewRun />} />
              <Route path="/runs/:id" element={<RunDetail />} />
              <Route path="/system" element={<System />} />
            </Routes>
          )}
        </main>
      </div>
    </SchemaContext.Provider>
  );
}
