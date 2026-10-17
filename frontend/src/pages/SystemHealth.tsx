import { useEffect, useState } from "react";
import { api, Health } from "../api";

// Control-room health: every part of the system at a glance, refreshed every 5 s. All measured on this machine.

const DOT: Record<string, string> = { ok: "live", warn: "busy", down: "red" };
const WORD: Record<string, string> = { ok: "All systems normal", warn: "Needs attention", down: "Something is down" };

function uptime(s: number) {
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
}

const ms = (v: number | null) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`);

export default function SystemHealth() {
  const [h, setH] = useState<Health | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const load = () => api.health().then((x) => { setH(x); setErr(null); }).catch((e) => setErr(String(e.message || e)));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">System health</h1>
          <p className="page-sub">Every part of Aequitas on this machine, checked every 5 seconds. Nothing leaves the station.</p>
        </div>
        {h && (
          <span className={"pill " + (h.status === "ok" ? "ok" : h.status === "down" ? "alert" : "")} style={{ marginLeft: "auto" }}>
            <span className={"dot " + DOT[h.status]} /> {WORD[h.status]} · up {uptime(h.uptime_s)}
          </span>
        )}
      </div>
      {err && <div className="card card-body" style={{ color: "var(--danger)" }}>Health check failed: {err}</div>}
      {!h && !err && <div className="empty">Checking…</div>}
      {h && (
        <>
          <div className="stat-strip">
            <div className="card stat">
              <div className="stat-label">Behind live (p50, 15 min)</div>
              <div className="stat-value mono">{h.pipeline.lag_s_p50 == null ? "–" : `${h.pipeline.lag_s_p50.toFixed(1)} s`}</div>
            </div>
            <div className="card stat">
              <div className="stat-label">Summarise · classify (p50)</div>
              <div className="stat-value mono">{ms(h.pipeline.summarize_ms_p50)} · {ms(h.pipeline.classify_ms_p50)}</div>
            </div>
            <div className="card stat">
              <div className="stat-label">Windows analysed (15 min)</div>
              <div className="stat-value mono">{h.pipeline.windows}{h.pipeline.failed ? <span style={{ color: "var(--warn)", fontSize: 13 }}> · {h.pipeline.failed} failed</span> : null}</div>
            </div>
            <div className="card stat">
              <div className="stat-label">GPU memory</div>
              <div className="stat-value mono">
                {h.gpu ? `${(h.gpu.memory_used_mb / 1024).toFixed(1)} / ${(h.gpu.memory_total_mb / 1024).toFixed(1)} GB` : "no GPU"}
              </div>
              {h.gpu && (
                <div className="meter" style={{ marginTop: 6 }}>
                  <i style={{ width: `${(100 * h.gpu.memory_used_mb) / h.gpu.memory_total_mb}%`, background: "var(--accent)" }} />
                </div>
              )}
            </div>
          </div>
          <section className="card">
            <div className="card-head">
              <span className="card-title">Checks</span>
              <span className="card-sub">last checked {new Date(h.checked_at).toLocaleTimeString("en-IN", { hour12: false })}</span>
            </div>
            <table className="queue">
              <tbody>
                {h.checks.map((c) => (
                  <tr key={c.name}>
                    <td style={{ width: 28 }}><span className={"dot " + DOT[c.status]} /></td>
                    <td style={{ width: 180, fontWeight: 600 }}>{c.name}</td>
                    <td>{c.detail}</td>
                    <td style={{ width: 70, textAlign: "right" }}><span className="badge status">{c.status}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
          <p className="note" style={{ marginTop: 10 }}>
            Model: {h.vlm.backend} · {h.vlm.model} · {h.pipeline.mode} mode at {h.pipeline.sample_fps} fps.
            {h.audit.ok && h.audit.head && <> Audit chain head <span className="mono">{h.audit.head.slice(0, 16)}…</span></>}
          </p>
        </>
      )}
    </main>
  );
}
