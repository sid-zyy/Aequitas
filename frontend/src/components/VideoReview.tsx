import { FormEvent, useEffect, useRef, useState } from "react";
import { BodyEvent, isIncident, label, RecordingDetail, VideoAnswer } from "../api";

// One video reviewed in depth: AI summary, player with a 4 s window timeline, key moments with keyframes that an
// officer confirms or dismisses, the full window log, and a chat that answers only about this video.
// Used by Bodycam review (officer recordings) and Active requests (citizen reports).

export const mmss = (s: number) => {
  const t = Math.max(0, Math.round(s));
  return `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(t % 60).padStart(2, "0")}`;
};
export const running = (s: string) => ["queued", "processing", "verifying", "summarising"].includes(s);

export type ReviewApi = {
  review: (eventId: number, decision: "confirmed" | "dismissed" | null) => Promise<BodyEvent>;
  summarize: () => Promise<{ summary: RecordingDetail["summary"]; summary_s: number | null }>;
  ask: (question: string, history: { q: string; a: string }[]) => Promise<VideoAnswer>;
};

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

type Turn = { q: string; a?: VideoAnswer; error?: string };

export function VideoChat({ ask, seek, disabled }: {
  ask: ReviewApi["ask"]; seek: (t: number) => void; disabled?: string;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }); }, [turns]);

  async function send(e: FormEvent) {
    e.preventDefault();
    const q = text.trim();
    if (!q || busy) return;
    setText("");
    setBusy(true);
    const history = turns.filter((t) => t.a).map((t) => ({ q: t.q, a: t.a!.answer }));
    setTurns((ts) => [...ts, { q }]);
    try {
      const a = await ask(q, history);
      setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? { ...t, a } : t)));
    } catch (err) {
      setTurns((ts) => ts.map((t, i) => (i === ts.length - 1 ? { ...t, error: (err as Error).message } : t)));
    } finally {
      setBusy(false);
    }
  }

  // "[01:12]" in an answer becomes a button that jumps the player there (only times inside the video)
  function answerText(a: VideoAnswer) {
    const ok = new Set(a.cited_s);
    return a.answer.split(/(\[\d{1,2}:\d{2}(?::\d{2})?\])/).map((part, i) => {
      const m = part.match(/^\[(\d{1,2}):(\d{2})\]$/);
      const s = m ? Number(m[1]) * 60 + Number(m[2]) : null;
      return s != null && ok.has(s)
        ? <button key={i} className="btn ghost mono cite" onClick={() => seek(s)}>▶ {part.slice(1, -1)}</button>
        : <span key={i}>{part}</span>;
    });
  }

  return (
    <div className="video-chat">
      <div className="section-label">Ask about this video · answers come only from its analysis</div>
      <div className="video-chat-log">
        {!turns.length && (
          <div className="note">e.g. "What happens before the fight?", "Is anyone holding a weapon?", "What does the citizen claim that the video does not show?"</div>
        )}
        {turns.map((t, i) => (
          <div key={i} className="video-chat-turn">
            <div className="chat-msg me"><div className="chat-bubble">{t.q}</div></div>
            {t.a && (
              <div className="chat-msg">
                <div className="chat-bubble">{answerText(t.a)}</div>
                <div className="note mono">{(t.a.ms / 1000).toFixed(2)} s · {t.a.model}</div>
              </div>
            )}
            {t.error && <div className="error">{t.error}</div>}
            {!t.a && !t.error && <div className="note">Thinking…</div>}
          </div>
        ))}
        <div ref={end} />
      </div>
      <form className="video-chat-input" onSubmit={send}>
        <input className="input" value={text} maxLength={500} disabled={!!disabled}
          onChange={(e) => setText(e.target.value)} placeholder={disabled ?? "Ask a question about this video…"} />
        <button className="btn navy" disabled={busy || !text.trim() || !!disabled}>Ask</button>
      </form>
    </div>
  );
}

export default function VideoReview({ d, setD, api, extra }: {
  d: RecordingDetail; setD: (d: RecordingDetail) => void; api: ReviewApi; extra?: React.ReactNode;
}) {
  const [now, setNow] = useState(0);
  const [showLog, setShowLog] = useState(false);
  const [summarising, setSummarising] = useState(false);
  const [sumError, setSumError] = useState<string | null>(null);
  const video = useRef<HTMLVideoElement>(null);

  const seek = (t: number) => {
    if (!video.current) return;
    video.current.currentTime = Math.max(0, t);
    video.current.play().catch(() => undefined);
  };
  async function review(ev: BodyEvent, decision: "confirmed" | "dismissed" | null) {
    const upd = await api.review(ev.id, decision);
    setD({ ...d, events: d.events.map((e) => (e.id === upd.id ? upd : e)), headline: upd.headline ?? d.headline });
  }
  async function resummarise() {
    setSummarising(true);
    setSumError(null);
    try {
      const r = await api.summarize();
      setD({ ...d, summary: r.summary, summary_s: r.summary_s });
    } catch (err) {
      setSumError((err as Error).message);
    } finally {
      setSummarising(false);
    }
  }

  const s = d.stages;
  const current = d.windows.find((w) => now >= w.t && now < w.end);
  const open = d.events.filter((e) => e.review == null).length;

  return (
    <div className="video-review">
      <div className="video-review-main">
        {d.error && <div className="error">{d.error}</div>}
        {d.summary ? (
          <div className="bc-summary">
            <div className="section-label">Summary of the video · AI draft</div>
            {d.headline && <p className="bc-headline">{d.headline}</p>}
            <p>{d.summary.overview}</p>
            {d.summary.key_points.length > 0 && (
              <ol>
                {d.summary.key_points.map((k, i) => (
                  <li key={i}><button className="btn ghost mono" onClick={() => seek(k.t)}>▶ {k.time}</button> {k.point}</li>
                ))}
              </ol>
            )}
            <div className="note" style={{ display: "flex", gap: 10, alignItems: "center" }}>
              Written from the 4 s summaries{d.summary_s != null ? ` in ${d.summary_s.toFixed(1)} s` : ""}.
              <button className="btn ghost" disabled={summarising} onClick={resummarise} style={{ marginLeft: "auto" }}>
                {summarising ? "Rewriting…" : "Rewrite with officer decisions"}
              </button>
            </div>
          </div>
        ) : d.status === "done" && (
          <div className="note">No summary yet. <button className="btn ghost" disabled={summarising} onClick={resummarise}>{summarising ? "Writing…" : "Write summary"}</button></div>
        )}
        {sumError && <div className="error">{sumError}</div>}
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
            <div className="stat-value">{d.windows.length} / {(s?.skipped_gate ?? 0) + (s?.skipped_static ?? 0)}</div></div>
          <div className="stat"><div className="stat-label">Per window: summary + label</div>
            <div className="stat-value">{s?.summarize_ms.mean != null ? `${(s.summarize_ms.mean / 1000).toFixed(2)} + ${((s.classify_ms.mean ?? 0) / 1000).toFixed(2)} s` : "–"}</div></div>
        </div>
        {running(d.status) && (
          <div className="note">Analysing… {Math.round(d.progress * 100)} % read{d.vlm_backlog ? `, ${d.vlm_backlog} windows waiting for the AI` : ""}.</div>
        )}
        {extra}
      </div>

      <div className="video-review-side">
        <div>
          <div className="section-label">Key moments · {d.events.length} found, {open} waiting for review</div>
          {!d.events.length && <div className="empty">{running(d.status) ? "None yet." : "The AI found nothing to review in this video."}</div>}
          <div style={{ display: "grid", gap: 8 }}>
            {d.events.map((ev) => (
              <Moment key={ev.id} ev={ev} active={now >= ev.t && now < ev.end} seek={() => seek(ev.t)} review={(x) => review(ev, x)} />
            ))}
          </div>
        </div>
        <VideoChat ask={api.ask} seek={seek}
          disabled={running(d.status) && !d.windows.length ? "Available once the first windows are analysed" : undefined} />
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
        <p className="note">The AI describes actions, never who people are. Nothing here is an incident until an officer confirms it.</p>
      </div>
    </div>
  );
}
