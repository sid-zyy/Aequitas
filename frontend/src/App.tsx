import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";
import { api, User } from "./api";
import Header from "./components/Header";
import Bodycam from "./pages/Bodycam";
import CitizenReport from "./pages/CitizenReport";
import Live from "./pages/Live";
import Login from "./pages/Login";
import Reports from "./pages/Reports";

export default function App() {
  const [user, setUser] = useState<User | null | undefined>(undefined); // undefined = still checking
  const loc = useLocation();

  useEffect(() => {
    api.me().then(setUser).catch(() => setUser(null));
    const out = () => setUser(null);
    window.addEventListener("aequitas:signed-out", out);
    return () => window.removeEventListener("aequitas:signed-out", out);
  }, []);

  // The citizen portal is public and has its own layout.
  if (loc.pathname.startsWith("/report")) return <CitizenReport />;
  if (user === undefined) return <div className="tricolour"><i /><i /><i /></div>;
  if (user === null) return <Login onSignedIn={setUser} />;

  return (
    <>
      <Header user={user} onSignOut={() => api.logout().finally(() => setUser(null))} />
      <Routes>
        <Route path="/" element={<Live />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/bodycam" element={<Bodycam />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}
