import { useEffect, useId, useState } from "react";
import { NavLink } from "react-router-dom";
import { api, openCitizenPortal, User } from "../api";

export function BrandMark({ size = 34 }: { size?: number }) {
  // A circular seal: navy and gold rings, the scales of justice (Aequitas: fairness), saffron and green laurel arcs.
  // Deliberately not the State Emblem or any police crest: their use is restricted by law.
  const id = useId().replace(/:/g, "");
  const ring = size >= 48; // the lettered ring is only legible at larger sizes
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 100 100" role="img" aria-label="Aequitas seal">
      <circle cx="50" cy="50" r="48" fill="#0b2a6f" stroke="#c9a349" strokeWidth="2.5" />
      {ring && (
        <>
          <path id={`seal-${id}`} d="M50,50 m-40,0 a40,40 0 1,1 80,0 a40,40 0 1,1 -80,0" fill="none" />
          <text fontSize="8.2" fontWeight="700" letterSpacing="1.6" fill="#f3e3b5" fontFamily="Georgia, 'Times New Roman', serif">
            <textPath href={`#seal-${id}`} startOffset="50%" textAnchor="middle">AEQUITAS ★ PUBLIC SAFETY ★ BENGALURU ★</textPath>
          </text>
        </>
      )}
      <circle cx="50" cy="50" r={ring ? 33 : 40} fill="#fff" stroke="#c9a349" strokeWidth="1.6" />
      <g transform={ring ? "translate(50 50) scale(0.82) translate(-50 -50)" : undefined}>
        <path d="M22 66 Q14 48 27 30" fill="none" stroke="#ff9933" strokeWidth="3" strokeLinecap="round" />
        <path d="M78 66 Q86 48 73 30" fill="none" stroke="#138808" strokeWidth="3" strokeLinecap="round" />
        <g stroke="#0b2a6f" strokeLinecap="round" fill="#0b2a6f">
          <line x1="50" y1="30" x2="50" y2="68" strokeWidth="3.4" />
          <circle cx="50" cy="27.5" r="3" stroke="none" />
          <line x1="33" y1="37" x2="67" y2="37" strokeWidth="3.4" />
          <path d="M33 37 L26 53 M33 37 L40 53 M67 37 L60 53 M67 37 L74 53" strokeWidth="1.5" fill="none" />
          <path d="M24.5 53 Q33 62 41.5 53 Z M58.5 53 Q67 62 75.5 53 Z" stroke="none" />
          <rect x="39" y="67" width="22" height="4.5" rx="1.5" stroke="none" />
        </g>
      </g>
    </svg>
  );
}

export function Tricolour() {
  return <div className="tricolour" aria-hidden><i /><i /><i /></div>;
}

function Clock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <div style={{ textAlign: "right" }}>
      <div className="clock mono">{now.toLocaleTimeString("en-IN", { hour12: false })}</div>
      <div className="brand-sub">{now.toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short" })} · IST</div>
    </div>
  );
}

function useOpenCount() {
  const [n, setN] = useState<number | null>(null);
  useEffect(() => {
    const load = () => api.activeReports().then((q) => setN(q.counts.received)).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);
  return n;
}

// The "On-premise AI" pill doubles as a hidden switch: clicking it pauses or resumes live CCTV analysis so the
// model isn't busy when nobody is watching. Only the dot shows it (grey when paused).
function AiSwitch() {
  const [paused, setPaused] = useState(false);
  useEffect(() => {
    const load = () => api.liveAi().then((r) => setPaused(r.paused)).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);
  const toggle = () => {
    setPaused(!paused);
    api.setLiveAi(!paused).then((r) => setPaused(r.paused)).catch(() => setPaused(paused));
  };
  return (
    <button type="button" className="pill pill-switch" onClick={toggle} aria-pressed={paused}>
      <span className={"dot " + (paused ? "" : "live")} /> On-premise AI · no cloud
    </button>
  );
}

export default function Header({ user, onSignOut }: { user: User; onSignOut: () => void }) {
  const fresh = useOpenCount();
  return (
    <>
      <Tricolour />
      <header className="topbar">
        <div className="brand">
          <BrandMark />
          <div>
            <div className="brand-name">AEQUITAS</div>
            <div className="brand-sub">Public Safety Command · Bengaluru <b>ಸುರಕ್ಷಿತ ಬೆಂಗಳೂರು</b></div>
          </div>
        </div>
        <nav className="tabs">
          <NavLink to="/" end className={({ isActive }) => "tab" + (isActive ? " active" : "")}>Live CCTV</NavLink>
          <NavLink to="/requests" className={({ isActive }) => "tab" + (isActive ? " active" : "")}>
            Active requests{!!fresh && <span className="count" title="new, not yet reviewed">{fresh}</span>}
          </NavLink>
          <NavLink to="/bodycam" className={({ isActive }) => "tab" + (isActive ? " active" : "")}>Bodycam review</NavLink>
          <NavLink to="/reports" className={({ isActive }) => "tab" + (isActive ? " active" : "")}>All reports</NavLink>
        </nav>
        <div className="topbar-right">
          <AiSwitch />
          <button className="btn ghost" onClick={openCitizenPortal} title="Opens the public citizen site in its own window">Citizen portal ↗</button>
          <Clock />
          <div className="user-chip">
            <div className="avatar">{user.username.slice(0, 2).toUpperCase()}</div>
            <div>
              <div style={{ color: "var(--text)", fontWeight: 600 }}>{user.username}</div>
              <div className="brand-sub" style={{ textTransform: "capitalize" }}>{user.role}</div>
            </div>
            <button className="btn ghost" onClick={onSignOut}>Sign out</button>
          </div>
        </div>
      </header>
    </>
  );
}
