import { useEffect, useMemo, useState } from "react";
import { api, isIncident, label, Locality, LiveState, timeOf } from "../api";
import AnalystChat from "../components/AnalystChat";
import CameraTile from "../components/CameraTile";

const SAVED = "aequitas.locality";

export default function Live() {
  const [localities, setLocalities] = useState<Locality[]>([]);
  const [selected, setSelected] = useState<string>(() => {
    try { return localStorage.getItem(SAVED) ?? ""; } catch { return ""; }
  });
  const [state, setState] = useState<LiveState | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The laptop webcam (source: browser) stays closed until asked for, so the page never pops a camera prompt.
  const [webcamOpen, setWebcamOpen] = useState(false);

  useEffect(() => {
    api.localities().then((l) => {
      setLocalities(l);
      setSelected((s) => (s && l.some((x) => x.id === s) ? s : l[0]?.id ?? ""));
    }).catch((e) => setError(String(e.message ?? e)));
  }, []);

  // Selecting a locality starts its analysis (one locality at a time fits one GPU), then poll.
  useEffect(() => {
    if (!selected) return;
    try { localStorage.setItem(SAVED, selected); } catch { /* private mode */ }
    let alive = true;
    setState(null);
    api.startLive(selected).then((s) => alive && setState(s)).catch((e) => setError(e.message));
    const t = setInterval(() => {
      api.liveState(selected).then((s) => alive && setState(s)).catch(() => {});
    }, 1500);
    return () => { alive = false; clearInterval(t); };
  }, [selected]);

  const stats = useMemo(() => {
    const cams = (state?.cameras ?? []).filter((c) => webcamOpen || !c.browser_camera);
    const latest = cams.map((c) => c.latest).filter(Boolean);
    const avg = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    return {
      alerts: cams.filter((c) => isIncident(c.latest?.label)).length,
      cams: cams.filter((c) => c.has_footage).length,
      window: avg(latest.map((w) => ((w!.summarize_ms ?? 0) + (w!.classify_ms ?? 0)) / 1000)),
      behind: avg(cams.map((c) => c.behind_live_s).filter((x): x is number => x != null)),
    };
  }, [state, webcamOpen]);

  const locName = state?.locality.name ?? localities.find((l) => l.id === selected)?.name ?? "";
  const cams = (state?.cameras ?? []).filter((c) => !c.browser_camera);
  const webcams = (state?.cameras ?? []).filter((c) => c.browser_camera);

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Live CCTV · {locName || "…"}</h1>
          <p className="page-sub">Multi-camera analysis: summary → classification, timed per window.</p>
        </div>
        <div className="toolbar" style={{ marginLeft: "auto" }}>
          <label className="label" htmlFor="loc" style={{ margin: 0 }}>Locality</label>
          <select id="loc" className="select locality-select" value={selected} onChange={(e) => setSelected(e.target.value)}>
            {localities.map((l) => (
              <option key={l.id} value={l.id}>{l.name} ({l.cameras.length} {l.cameras.length === 1 ? "camera" : "cameras"})</option>
            ))}
          </select>
          {state && (
            <span className={"pill " + (state.running ? "ok" : "")}>
              <span className={"dot " + (state.running ? "live" : "")} />
              {state.running ? `AI running · ${state.model.name}` : "AI not running"}
            </span>
          )}
          {stats.alerts > 0 && <span className="pill alert">⚠ {stats.alerts} active {stats.alerts === 1 ? "alert" : "alerts"}</span>}
        </div>
      </div>

      {(error || state?.error) && <p className="error">{error ?? state?.error}</p>}

      <div className="live-layout">
        <div>
          <div className="stat-strip">
            <div className="card stat"><div className="stat-value mono">{stats.cams}</div><div className="stat-label">Cameras live</div></div>
            <div className="card stat"><div className="stat-value mono" style={{ color: stats.alerts ? "var(--danger)" : undefined }}>{stats.alerts}</div><div className="stat-label">Active alerts</div></div>
            <div className="card stat"><div className="stat-value mono">{stats.window != null ? `${stats.window.toFixed(1)} s` : "–"}</div><div className="stat-label">AI time / window</div></div>
            <div className="card stat"><div className="stat-value mono">{stats.behind != null ? `${stats.behind.toFixed(1)} s` : "–"}</div><div className="stat-label">Behind live (avg)</div></div>
          </div>
          <div className={"cam-grid" + (cams.length === 3 || cams.length > 4 ? " three" : "")}>
            {cams.map((c) => <CameraTile key={c.id} cam={c} running={!!state?.running} />)}
          </div>
          {!cams.length && !error && <div className="card empty">Loading cameras…</div>}
          {webcams.length > 0 && (
            <div className="webcam-drawer">
              <button className="btn" onClick={() => setWebcamOpen((o) => !o)} aria-expanded={webcamOpen}>
                {webcamOpen ? "▾ Hide laptop webcam" : "▸ Laptop webcam (off)"}
              </button>
              {webcamOpen && (
                <div className="cam-grid">
                  {webcams.map((c) => <CameraTile key={c.id} cam={c} running={!!state?.running} />)}
                </div>
              )}
            </div>
          )}
        </div>

        <aside className="sidebar">
          <section className="card">
            <div className="card-head">
              <span className="card-title">⚠ Concerns feed</span>
              <span className="card-sub">{locName} · unconfirmed until an officer checks</span>
            </div>
            <div className="feed">
              {!state?.concerns.length && <div className="empty">No concerns yet.</div>}
              {state?.concerns.map((ev) => (
                <div key={ev.id} className="concern">
                  <img src={ev.keyframe_url} alt="" loading="lazy" />
                  <div style={{ minWidth: 0 }}>
                    <div className="concern-meta">
                      <span className="badge incident">{label(ev.category)}</span>
                      <span className={"badge " + (ev.status === "confirmed" ? "status" : "unverified")}>{ev.status}</span>
                      <span className="mono">{timeOf(ev.start_ts)}</span>
                    </div>
                    <div style={{ fontWeight: 600, marginTop: 3 }}>{ev.camera_name}</div>
                    <p>{ev.summary}</p>
                  </div>
                </div>
              ))}
            </div>
          </section>
          <AnalystChat localityName={locName} />
        </aside>
      </div>
    </main>
  );
}
