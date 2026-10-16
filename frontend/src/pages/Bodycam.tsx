import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, Recording, RecordingDetail } from "../api";
import VideoReview, { mmss, running } from "../components/VideoReview";

const day = (ms: number) => new Date(ms).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });

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

export default function Bodycam() {
  const [list, setList] = useState<Recording[]>([]);
  const [params, setParams] = useSearchParams();
  const sel = params.get("id");
  const [d, setD] = useState<RecordingDetail | null>(null);
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
            <div className="card-body">
              <VideoReview d={d} setD={setD} api={{
                review: (eventId, decision) => api.reviewMoment(d.id, eventId, decision),
                summarize: () => api.summarizeRecording(d.id),
                ask: (q, history) => api.askRecording(d.id, q, history),
              }} />
            </div>
          </section>
        ) : (
          <section className="card"><div className="empty">Upload a recording or pick one from the list.</div></section>
        )}
      </div>
    </main>
  );
}
