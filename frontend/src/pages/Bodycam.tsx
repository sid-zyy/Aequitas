import { FormEvent, useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, BodyEvent, isIncident, label, Recording, RecordingDetail } from "../api";

const mmss = (s: number) => {
  const t = Math.max(0, Math.round(s));
  return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
};
const day = (ms: number) => new Date(ms).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });
const running = (s: string) => s === "queued" || s === "processing" || s === "verifying";

function UploadForm({ onDone }: { onDone: (r: Recording) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [officer, setOfficer] = useState("");
  const [when, setWhen] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    const f = new FormData();
    f.append("file", file);
    f.append("officer", officer);
    if (when) f.append("recorded_at", when);
    try {
      onDone(await api.uploadRecording(f));
      setFile(null);
      (e.target as HTMLFormElement).reset();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card-body" onSubmit={submit} style={{ borderBottom: "1px solid var(--line)" }}>
      <div className="form-row">
        <label className="label">Recording</label>
        <input className="input" type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
        <div className="form-row">
          <label className="label">Officer / unit</label>
          <input className="input" value={officer} maxLength={80} onChange={(e) => setOfficer(e.target.value)} placeholder="e.g. PC 4412, Traffic East" />
        </div>
        <div className="form-row">
          <label className="label">Recorded at</label>
          <input className="input" type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      <button className="btn primary" disabled={!file || busy}>{busy ? "Uploading…" : "Upload and analyse"}</button>
    </form>
  );
}

function Timeline({ d, now, seek }: { d: RecordingDetail; now: number; seek: (t: number) => void }) {
  const len = Math.max(d.duration_s, 0.001);
  const pct = (t: number) => `${Math.min(100, (t / len) * 100)}%`;
  return (
    <div>
      <div className="bc-timeline" onClick={(e) => {
        const r = e.currentTarget.getBoundingClientRect();
        seek(((e.clientX - r.left) / r.width) * len);
      }}>
        {d.windows.map((w, i) => (
          <i key={i} className={w.label == null ? "fail" : isIncident(w.label) ? "inc" : "ok"}
            style={{ left: pct(w.t), width: pct(w.end - w.t) }} title={`${mmss(w.t)} ${label(w.label ?? "failed")}`} />
        ))}
        {d.events.map((ev) => (
          <b key={ev.id} className={"bc-mark" + (ev.review === "dismissed" ? " dim" : "")} style={{ left: pct(ev.t) }} title={label(ev.category)} />
        ))}
        <span className="bc-head" style={{ left: pct(now) }} />
      </div>
      <div className="bc-legend note">
        <span><i className="ok" /> nothing found</span><span><i className="inc" /> incident</span>
        <span><i className="gap" /> not analysed (looked ordinary to the quick filter, or still running)</span>
        <span style={{ marginLeft: "auto" }} className="mono">{mmss(now)} / {mmss(d.duration_s)}</span>
      </div>
    </div>
  );
}

function Moment({ ev, active, seek, review }: {
  ev: BodyEvent; active: boolean; seek: () => void; review: (d: "confirmed" | "dismissed" | null) => void;
}) {
  return (
    <div className={"bc-moment" + (active ? " sel" : "") + (ev.review === "dismissed" ? " dim" : "")}>
      <img src={ev.keyframe_url} alt="" onClick={seek} />
      <div style={{ display: "grid", gap: 4, minWidth: 0 }}>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <button className="btn ghost mono" onClick={seek}>▶ {mmss(ev.t)}</button>
          <span className="badge incident">{label(ev.category)}</span>
          <span className={"badge " + (ev.status === "confirmed" ? "status" : "unverified")}>AI: {ev.status}</span>
          {ev.review && <span className="badge status">{ev.review} by {ev.reviewed_by}</span>}
        </div>
        <div style={{ color: "#cfdbe3" }}>{ev.summary}</div>
        <div style={{ display: "flex", gap: 8 }}>
          {ev.review !== "confirmed" && <button className="btn" onClick={() => review("confirmed")}>Confirm</button>}
          {ev.review !== "dismissed" && <button className="btn ghost" onClick={() => review("dismissed")}>Dismiss</button>}
          {ev.review && <button className="btn ghost" onClick={() => review(null)}>Undo</button>}
        </div>
      </div>
    </div>
  );
}

export default function Bodycam() {
  const [list, setList] = useState<Recording[]>([]);
  const [params, setParams] = useSearchParams();
  const sel = params.get("id");
  const [d, setD] = useState<RecordingDetail | null>(null);
  const [now, setNow] = useState(0);
  const [showLog, setShowLog] = useState(false);
  const video = useRef<HTMLVideoElement>(null);
  const choose = (id: string) => setParams({ id });

  const loadList = () => api.recordings().then((r) => { setList(r); if (!sel && r[0]) choose(r[0].id); });
  useEffect(() => { loadList(); }, []); // eslint-disable-line
  useEffect(() => {
    if (!sel) return;
    let stop = false;
    const tick = () => api.recording(sel).then((x) => { if (!stop) setD(x); return x; });
    tick();
    const t = setInterval(() => {
      tick().then((x) => { if (!running(x.status)) clearInterval(t); loadList(); });
    }, 3000);
    return () => { stop = true; clearInterval(t); };
  }, [sel]); // eslint-disable-line

  const seek = (t: number) => {
    if (!video.current) return;
    video.current.currentTime = Math.max(0, t);
    video.current.play().catch(() => undefined);
  };
  async function review(ev: BodyEvent, decision: "confirmed" | "dismissed" | null) {
    if (!d) return;
    const upd = await api.reviewMoment(d.id, ev.id, decision);
    setD({ ...d, events: d.events.map((e) => (e.id === upd.id ? upd : e)) });
  }

  const s = d?.stages;
  const analysed = d ? d.windows.length : 0;
  const current = d?.windows.find((w) => now >= w.t && now < w.end);
  const open = d ? d.events.filter((e) => e.review == null).length : 0;

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Bodycam review</h1>
          <p className="page-sub">An officer's recording, summarised every 4 seconds. The AI marks key moments; an officer confirms or dismisses each one.</p>
        </div>
      </div>
      <div className="split">
        <section className="card">
          <div className="card-head"><span className="card-title">Recordings</span><span className="card-sub">{list.length}</span></div>
          <UploadForm onDone={(r) => { loadList(); choose(r.id); }} />
          {!list.length && <div className="empty">No recordings yet.</div>}
          {list.map((r) => (
            <div key={r.id} className={"report-row" + (sel === r.id ? " sel" : "")} onClick={() => choose(r.id)}>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <b style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.officer || "Unnamed officer"}</b>
                <span className="badge status">{r.status}</span>
                {r.events_found > 0 && <span className="badge incident">{r.events_found} moments</span>}
                <span className="note" style={{ marginLeft: "auto" }}>{mmss(r.duration_s)}</span>
              </div>
              <div className="note" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.filename} · {day(r.start_ts)}</div>
              {running(r.status) && <div className="meter"><i style={{ width: `${Math.round(r.progress * 100)}%` }} /></div>}
            </div>
          ))}
        </section>

        {d ? (
          <section className="card">
            <div className="card-head">
              <span className="card-title">{d.officer || "Unnamed officer"} · {d.filename}</span>
              <span className="card-sub">{day(d.start_ts)}</span>
              <a className="btn" style={{ marginLeft: "auto" }} href={`/api/bodycam/${d.id}/report`}>Export report</a>
            </div>
            <div className="card-body" style={{ display: "grid", gap: 14 }}>
              {d.error && <div className="error">{d.error}</div>}
              <video ref={video} className="player" src={d.video_url} controls onTimeUpdate={(e) => setNow(e.currentTarget.currentTime)} />
              <Timeline d={d} now={now} seek={seek} />
              {current?.summary && (
                <div className="summary"><span className="note mono">{mmss(current.t)}</span> {current.summary}</div>
              )}
              <div className="stat-strip">
                <div className="stat"><div className="stat-label">Length</div><div className="stat-value">{mmss(d.duration_s)}</div></div>
                <div className="stat"><div className="stat-label">Analysed in</div>
                  <div className="stat-value">{d.total_s != null ? `${d.total_s.toFixed(1)} s` : "–"}</div></div>
                <div className="stat"><div className="stat-label">Windows: AI / skipped</div>
                  <div className="stat-value">{analysed} / {(s?.skipped_gate ?? 0) + (s?.skipped_static ?? 0)}</div></div>
                <div className="stat"><div className="stat-label">Per window: summary + label</div>
                  <div className="stat-value">{s?.summarize_ms.mean != null ? `${(s.summarize_ms.mean / 1000).toFixed(2)} + ${((s.classify_ms.mean ?? 0) / 1000).toFixed(2)} s` : "–"}</div></div>
              </div>
              {running(d.status) && (
                <div className="note">Analysing… {Math.round(d.progress * 100)} % read{d.vlm_backlog ? `, ${d.vlm_backlog} windows waiting for the AI` : ""}.</div>
              )}

              <div>
                <div className="section-label">Key moments · {d.events.length} found, {open} waiting for review</div>
                {!d.events.length && <div className="empty">{running(d.status) ? "None yet." : "The AI found nothing to review in this recording."}</div>}
                <div style={{ display: "grid", gap: 8 }}>
                  {d.events.map((ev) => (
                    <Moment key={ev.id} ev={ev} active={now >= ev.t && now < ev.end} seek={() => seek(ev.t)} review={(x) => review(ev, x)} />
                  ))}
                </div>
              </div>

              <div>
                <button className="btn ghost" onClick={() => setShowLog(!showLog)}>{showLog ? "▾" : "▸"} Full timeline ({d.windows.length} windows)</button>
                {showLog && (
                  <div className="bc-log">
                    {d.windows.map((w, i) => (
                      <div key={i} className={current === w ? "sel" : ""} onClick={() => seek(w.t)}>
                        <span className="mono note">{mmss(w.t)}</span>
                        <span className={"badge " + (w.label == null ? "status" : isIncident(w.label) ? "incident" : "none")}>{label(w.label ?? "failed")}</span>
                        <span>{w.summary}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <p className="note">The AI describes actions, never who people are. Nothing in this recording is an incident until an officer confirms it.</p>
            </div>
          </section>
        ) : (
          <section className="card"><div className="empty">Upload a recording or pick one from the list.</div></section>
        )}
      </div>
    </main>
  );
}
