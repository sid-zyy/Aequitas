import { useEffect, useState } from "react";
import { NavLink } from "react-router-dom";
import { User } from "../api";

export function BrandMark({ size = 34 }: { size?: number }) {
  // A shield with a saffron/green edge: public-safety cue without using any official emblem.
  return (
    <div className="brand-mark" style={{ width: size, height: size }}>
      <svg width={size * 0.6} height={size * 0.6} viewBox="0 0 24 24" fill="none" aria-hidden>
        <path d="M12 2.5 4 5.5v6c0 5 3.4 8.6 8 10 4.6-1.4 8-5 8-10v-6l-8-3Z" fill="#0e1a4d" stroke="#fff" strokeWidth="1.4" />
        <path d="M8 12.2 11 15l5-6" stroke="#2fd3e8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 5.5 12 2.5" stroke="#ff9933" strokeWidth="1.6" strokeLinecap="round" />
        <path d="M12 21.5c4.6-1.4 8-5 8-10" stroke="#1e9e3a" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    </div>
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

export default function Header({ user, onSignOut }: { user: User; onSignOut: () => void }) {
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
          <NavLink to="/bodycam" className={({ isActive }) => "tab" + (isActive ? " active" : "")}>Bodycam review</NavLink>
          <NavLink to="/reports" className={({ isActive }) => "tab" + (isActive ? " active" : "")}>Citizen reports</NavLink>
        </nav>
        <div className="topbar-right">
          <span className="pill"><span className="dot live" /> On-premise AI · no cloud</span>
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
