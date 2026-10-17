import { FormEvent, useEffect, useRef, useState } from "react";
import { askStream, EventItem, label, timeOf } from "../api";

type Msg = {
  role: "user" | "bot"; text: string; head?: string[]; events?: EventItem[];
  firstWord?: number | null; total?: number | null; dropped?: boolean;
};

export default function AnalystChat({ localityName }: { localityName: string }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const log = useRef<HTMLDivElement>(null);
  useEffect(() => { log.current?.scrollTo(0, log.current.scrollHeight); }, [msgs]);

  const suggestions = [
    `Any incidents in ${localityName} in the last hour?`,
    "Show all arrests today",
    "Was there any fire today?",
    "How many fights this week?",
  ];

  async function ask(question: string) {
    if (!question.trim() || busy) return;
    setBusy(true);
    setQ("");
    const t0 = performance.now();
    setMsgs((m) => [...m, { role: "user", text: question }, { role: "bot", text: "" }]);
    const patch = (f: (m: Msg) => Msg) => setMsgs((all) => [...all.slice(0, -1), f(all[all.length - 1])]);
    try {
      await askStream(question, (e) => {
        if (e.type === "retrieved") patch((m) => ({ ...m, head: e.head, events: e.events }));
        if (e.type === "token")
          patch((m) => ({ ...m, text: m.text + e.text, firstWord: m.firstWord ?? performance.now() - t0 }));
        if (e.type === "summary") patch((m) => ({ ...m, text: e.summary ?? "", dropped: e.summary == null && !!m.text }));
        if (e.type === "done") patch((m) => ({ ...m, total: performance.now() - t0 }));
      });
    } catch {
      patch((m) => ({ ...m, text: "Could not reach the analyst model." }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card chat">
      <div className="card-head">
        <span className="card-title">💬 Analyst chat</span>
        <span className="card-sub">answers cite events [#id]</span>
      </div>
      <div className="chat-log" ref={log}>
        {msgs.length === 0 && <div className="empty">Ask about events across all cameras.</div>}
        {msgs.map((m, i) => (
          <div key={i} className={"msg " + m.role}>
            {m.role === "bot" && m.head?.map((h) => <div key={h}>{h}</div>)}
            {m.text}
            {m.role === "bot" && m.dropped && <div className="note">Summary removed: it cited events that were not found.</div>}
            {m.role === "bot" && !!m.events?.length && (
              <div className="refs">
                {m.events.slice(0, 5).map((ev) => (
                  <span key={ev.id}>[#{ev.id}] {timeOf(ev.start_ts)} · {ev.camera_name}{ev.area ? `, ${ev.area}` : ""} · <b>{label(ev.category)}</b></span>
                ))}
              </div>
            )}
            {m.role === "bot" && m.total != null && (
              <div className="timings" style={{ marginTop: 6 }}>
                {m.firstWord != null && <span className="chip accent">first word {(m.firstWord / 1000).toFixed(2)} s</span>}
                <span className="chip">full answer {(m.total / 1000).toFixed(2)} s</span>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="suggest">
        {suggestions.map((s) => <button key={s} onClick={() => ask(s)} disabled={busy}>{s}</button>)}
      </div>
      <form className="chat-input" onSubmit={(e: FormEvent) => { e.preventDefault(); ask(q); }}>
        <input className="input" placeholder="Ask a question…" value={q} onChange={(e) => setQ(e.target.value)} />
        <button className="btn primary" disabled={busy || !q.trim()}>{busy ? "…" : "Ask"}</button>
      </form>
    </section>
  );
}
