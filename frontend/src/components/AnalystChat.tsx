import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { askStream, ChatContext, ChatIndex, chatIndex, EventItem, label, Moment, timeOf } from "../api";

type Mode = "filter" | "hybrid" | "detail" | "similar";
type Timing = { parse?: number; embed?: number | null; vector?: number | null; query?: number | null; firstWord?: number; total?: number };
type Msg = {
  role: "user" | "bot"; text: string; head?: string[]; events?: EventItem[]; moments?: Moment[];
  total?: number; mode?: Mode; kept?: string[]; suggestions?: string[]; timing?: Timing;
  dropped?: boolean; error?: boolean; stopped?: boolean; pending?: boolean; weak?: boolean; earlier?: boolean;
};

const MODE_INFO: Record<Mode, { icon: string; name: string; hint: string }> = {
  filter: { icon: "⛁", name: "Filter search", hint: "SQL over category, place, time and status" },
  hybrid: { icon: "🧠", name: "Vector + keyword search", hint: "Qdrant by meaning + BM25 by words, inside the SQL filter" },
  detail: { icon: "📄", name: "Event detail", hint: "the event and what its camera saw around it" },
  similar: { icon: "≈", name: "Similar events", hint: "nearest summaries in the vector database" },
};

const STARTERS = (loc: string) => [
  `Any incidents in ${loc} in the last hour?`,
  "Was anyone carrying something heavy?",
  "Did anyone fall near a bus stop?",
  "Show people running",
  "How many fights this week?",
];

const ms = (v: number | null | undefined) => (v == null ? null : v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`);

function contextLabel(c: ChatContext | null): string | null {
  if (!c?.filter) return null;
  const f = c.filter as { categories?: string[]; camera_ids?: string[]; start?: string | null; end?: string | null; statuses?: string[]; text?: string | null };
  const parts = [
    ...(f.categories ?? []).map(label),
    ...(f.camera_ids?.length ? [`${f.camera_ids.length} camera${f.camera_ids.length > 1 ? "s" : ""}`] : []),
    ...(f.start || f.end ? ["time range"] : []),
    ...((f.statuses?.length ?? 1) > 1 ? ["all statuses"] : []),
    ...(f.text ? [`“${f.text}”`] : []),
  ];
  if (c.mode === "detail" || c.mode === "similar") parts.unshift(c.event_ids.map((i) => `#${i}`).slice(0, 3).join(" "));
  return parts.length ? parts.join(" · ") : "all events";
}

/** Summary text with [#12] / [W7] citations as buttons that jump to the card they cite. */
function Cited({ text, onCite }: { text: string; onCite: (ref: string) => void }) {
  const parts = text.split(/(\[#\d+\]|\[W\d+\])/g);
  return (
    <>
      {parts.map((p, i) =>
        /^\[(#|W)\d+\]$/.test(p)
          ? <button key={i} className="cite" onClick={() => onCite(p.slice(1, -1))} title="Show what this cites">{p.slice(1, -1)}</button>
          : <span key={i}>{p}</span>)}
    </>
  );
}

function Relevance({ v, matched }: { v?: number | null; matched?: string | null }) {
  if (v == null && !matched) return null;
  const pct = v == null ? 0 : Math.max(0, Math.min(1, (v - 0.4) / 0.4));
  return (
    <span className="rel" title={`cosine similarity ${v?.toFixed(3) ?? "–"}${matched ? ` · matched by ${matched}` : ""}`}>
      {v != null && <span className="rel-bar"><span style={{ width: `${pct * 100}%` }} /></span>}
      {matched && <span className="rel-how">{matched.replace("+", " + ")}</span>}
    </span>
  );
}

export default function AnalystChat({ localityName }: { localityName: string }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [ctx, setCtx] = useState<ChatContext | null>(null);
  const [index, setIndex] = useState<ChatIndex | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [video, setVideo] = useState<{ url: string; at: number; title: string } | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  const [showAll, setShowAll] = useState<Record<number, boolean>>({});
  const log = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);
  const history = useRef<string[]>([]);
  const histPos = useRef(-1);

  useEffect(() => { log.current?.scrollTo({ top: log.current.scrollHeight, behavior: "smooth" }); }, [msgs.length, busy]);
  useEffect(() => {
    let alive = true;
    const load = () => chatIndex().then((s) => alive && setIndex(s)).catch(() => {});
    load();
    const t = setInterval(load, 15000);
    return () => { alive = false; clearInterval(t); };
  }, []);
  useEffect(() => {
    if (!expanded) return;
    const esc = (e: globalThis.KeyboardEvent) => e.key === "Escape" && !video && setExpanded(false);
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [expanded, video]);

  async function ask(question: string) {
    question = question.trim();
    if (!question || busy) return;
    history.current = [question, ...history.current.filter((h) => h !== question)].slice(0, 30);
    histPos.current = -1;
    setBusy(true);
    setQ("");
    const t0 = performance.now();
    const ac = new AbortController();
    abort.current = ac;
    setMsgs((m) => [...m, { role: "user", text: question }, { role: "bot", text: "", pending: true }]);
    const patch = (f: (m: Msg) => Msg) => setMsgs((all) => [...all.slice(0, -1), f(all[all.length - 1])]);
    try {
      await askStream(question, (e) => {
        if (e.type === "retrieved") {
          patch((m) => ({
            ...m, pending: false, head: e.head, events: e.events, moments: e.moments, total: e.total, mode: e.mode,
            kept: e.kept, suggestions: e.suggestions, weak: e.weak, earlier: e.earlier,
            timing: { parse: e.parse_ms, embed: e.embed_ms, vector: e.vector_ms, query: e.query_ms },
          }));
          setCtx(e.context);
        }
        if (e.type === "token")
          patch((m) => ({ ...m, text: m.text + e.text, timing: { ...m.timing, firstWord: m.timing?.firstWord ?? performance.now() - t0 } }));
        if (e.type === "summary") patch((m) => ({ ...m, text: e.summary ?? "", dropped: e.summary == null && !!m.text }));
        if (e.type === "done") patch((m) => ({ ...m, timing: { ...m.timing, total: performance.now() - t0 } }));
      }, ctx, ac.signal);
    } catch (err) {
      if ((err as Error).name === "AbortError") patch((m) => ({ ...m, pending: false, stopped: true, timing: { ...m.timing, total: performance.now() - t0 } }));
      else patch((m) => ({ ...m, pending: false, error: true, text: "Could not reach the analyst model." }));
    } finally {
      abort.current = null;
      setBusy(false);
    }
  }

  function cite(ref: string) {  // ref = "<message index>-#12" or "<message index>-W7"
    setFlash(ref);
    setShowAll((s) => ({ ...s, [Number(ref.split("-")[0])]: true }));  // the cited card may be folded away
    setTimeout(() => document.getElementById(`cite-${ref}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 30);
    setTimeout(() => setFlash((f) => (f === ref ? null : f)), 1600);
  }

  function onKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "ArrowUp" && history.current.length && (!q || histPos.current >= 0)) {
      e.preventDefault();
      histPos.current = Math.min(histPos.current + 1, history.current.length - 1);
      setQ(history.current[histPos.current]);
    } else if (e.key === "ArrowDown" && histPos.current >= 0) {
      e.preventDefault();
      histPos.current -= 1;
      setQ(histPos.current >= 0 ? history.current[histPos.current] : "");
    }
  }

  function reset() {
    abort.current?.abort();
    setMsgs([]);
    setCtx(null);
    setShowAll({});
  }

  const lastBot = msgs.length - 1;
  const ctxText = contextLabel(ctx);

  return (
    <>
      {expanded && <div className="chat-scrim" onClick={() => setExpanded(false)} />}
      <section className={"card chat" + (expanded ? " expanded" : "")} aria-label="Analyst chat">
        <div className="card-head">
          <span className="card-title">💬 Analyst chat</span>
          <span className="card-sub chat-index" title={index?.ready ? `${index.engine} (${index.mode}) · ${index.embedder}, ${index.dim}-d · last sync ${ms(index.last_sync_ms) ?? "–"}` : "vector index loading or off"}>
            {index?.ready
              ? <>🧠 Qdrant · {(index.events ?? 0) + (index.windows ?? 0)} summaries</>
              : index?.enabled === false ? "keyword search" : "vector index loading…"}
          </span>
          <span className="chat-tools">
            {msgs.length > 0 && <button className="btn ghost sm" onClick={reset} title="Start a new conversation">New</button>}
            <button className="btn ghost sm" onClick={() => setExpanded((x) => !x)} title={expanded ? "Back to the sidebar (Esc)" : "Open a larger chat"}>
              {expanded ? "⤡" : "⤢"}
            </button>
          </span>
        </div>

        <div className="chat-log" ref={log}>
          {msgs.length === 0 && (
            <div className="chat-empty">
              <div>Ask about anything the cameras saw, not only incidents.</div>
              <div className="note">Questions are matched by meaning in a vector database of every summary, inside your filters for place, time and type. Answers cite events <b>#id</b> and camera moments <b>W id</b>.</div>
            </div>
          )}
          {msgs.map((m, i) => m.role === "user" ? (
            <div key={i} className="msg user">{m.text}</div>
          ) : (
            <div key={i} className={"msg bot" + (m.error ? " err" : "") + (m.weak ? " weak" : "")}>
              {m.pending && <div className="typing" aria-label="searching"><span /><span /><span /></div>}
              {m.mode && (
                <div className="mode-line">
                  <span className={"mode-badge " + m.mode} title={MODE_INFO[m.mode].hint}>{MODE_INFO[m.mode].icon} {MODE_INFO[m.mode].name}</span>
                  {m.earlier && <span className="weak-badge" title="Nothing in the time you asked about; these are the latest before or after it">outside your time range</span>}
                  {m.weak && <span className="weak-badge" title="Below the close-match bar (chat.vector.strong_score)">weak matches only</span>}
                  {!!m.kept?.length && <span className="note">kept from before: {m.kept.join(", ")}</span>}
                </div>
              )}
              {m.head?.filter((h) => !h.startsWith("Following up")).map((h) => <div key={h} className="head-line">{h}</div>)}
              {m.text && <div className="answer"><Cited text={m.text} onCite={(r) => cite(`${i}-${r}`)} /></div>}
              {m.dropped && <div className="note">Summary removed: it cited events that were not found.</div>}
              {m.stopped && <div className="note">Stopped.</div>}

              {!!m.events?.length && (
                <div className="ev-list">
                  {(showAll[i] ? m.events : m.events.slice(0, 4)).map((ev) => (
                    <div key={ev.id} id={`cite-${i}-#${ev.id}`} className={"ev-card" + (flash === `${i}-#${ev.id}` ? " flash" : "")}>
                      <img src={ev.keyframe_url} alt="" loading="lazy" onError={(e) => ((e.target as HTMLImageElement).style.visibility = "hidden")} />
                      <div className="ev-main">
                        <div className="ev-meta">
                          <b className="mono">#{ev.id}</b>
                          <span className="badge incident">{label(ev.subtype || ev.category)}</span>
                          {ev.status !== "confirmed" && <span className="badge unverified">{ev.status}</span>}
                          <span className="mono">{timeOf(ev.start_ts)}</span>
                          <Relevance v={ev.relevance} matched={ev.matched} />
                        </div>
                        <div className="ev-place">{ev.camera_name}{ev.area ? `, ${ev.area}` : ""}</div>
                        {ev.summary && <p className="ev-sum">{ev.summary}</p>}
                        <div className="ev-actions">
                          {m.mode !== "detail" && <button onClick={() => ask(`Tell me more about #${ev.id}`)} disabled={busy}>Details</button>}
                          {index?.ready && <button onClick={() => ask(`Find events similar to #${ev.id}`)} disabled={busy}>Similar</button>}
                          {ev.video && <button onClick={() => setVideo({ url: ev.video!.url, at: ev.video!.offset_s, title: `#${ev.id} · ${ev.camera_name}` })}>▶ Play</button>}
                        </div>
                      </div>
                    </div>
                  ))}
                  {m.events.length > 4 && (
                    <button className="more" onClick={() => setShowAll((s) => ({ ...s, [i]: !s[i] }))}>
                      {showAll[i] ? "Show fewer" : `Show all ${m.events.length}${m.total && m.total > m.events.length ? ` of ${m.total}` : ""}`}
                    </button>
                  )}
                </div>
              )}

              {!!m.moments?.length && (
                <div className="moments">
                  <div className="section-label">{m.mode === "detail" ? "What the camera saw around it" : "Camera moments (not incidents)"}</div>
                  {m.moments.map((w) => (
                    <div key={w.id} id={`cite-${i}-W${w.id}`} className={"moment" + (flash === `${i}-W${w.id}` ? " flash" : "")}>
                      <span className="mono t">W{w.id} · {timeOf(w.start_ts)}</span>
                      <span className="m-body">
                        <span className="m-place">{w.camera_name}{w.label && w.label !== "none" ? ` · ${label(w.label)}` : ""}</span>
                        {w.summary}
                      </span>
                      <span className="m-side">
                        <Relevance v={w.relevance} />
                        {w.video && <button className="play" title="Play this moment" onClick={() => setVideo({ url: w.video!.url, at: w.video!.offset_s, title: `W${w.id} · ${w.camera_name}` })}>▶</button>}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {m.timing?.total != null && (
                <div className="timings" style={{ marginTop: 8 }} title="Where the time went">
                  {m.timing.parse != null && m.timing.parse > 0 && <span className="chip">parse {ms(m.timing.parse)}</span>}
                  {m.timing.embed != null && <span className="chip">embed {ms(m.timing.embed)}</span>}
                  {m.timing.vector != null && <span className="chip">vector search {ms(m.timing.vector)}</span>}
                  {m.timing.query != null && <span className="chip">SQL {ms(m.timing.query)}</span>}
                  {m.timing.firstWord != null && <span className="chip accent">first word {ms(m.timing.firstWord)}</span>}
                  <span className="chip saffron">total {ms(m.timing.total)}</span>
                </div>
              )}

              {i === lastBot && !busy && !!m.suggestions?.length && (
                <div className="follow">
                  {m.suggestions.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}
                </div>
              )}
            </div>
          ))}
        </div>

        {msgs.length === 0 && (
          <div className="suggest">
            {STARTERS(localityName).map((s) => <button key={s} onClick={() => ask(s)} disabled={busy}>{s}</button>)}
          </div>
        )}
        {ctxText && !busy && (
          <div className="ctx-pill" title="Follow-up questions build on this. Clear it to start fresh.">
            <span>↪ Follow-ups use: {ctxText}</span>
            <button onClick={() => setCtx(null)} aria-label="Clear follow-up context">✕</button>
          </div>
        )}
        <form className="chat-input" onSubmit={(e: FormEvent) => { e.preventDefault(); ask(q); }}>
          <input className="input" placeholder={ctx ? "Ask a follow-up, e.g. “and yesterday?”" : "Ask a question…"} value={q}
                 onChange={(e) => { setQ(e.target.value); histPos.current = -1; }} onKeyDown={onKey} aria-label="Question" />
          {busy
            ? <button type="button" className="btn" onClick={() => abort.current?.abort()}>Stop</button>
            : <button className="btn primary" disabled={!q.trim()}>Ask</button>}
        </form>
      </section>

      {video && (
        <div className="video-modal" onClick={() => setVideo(null)}>
          <div className="video-box" onClick={(e) => e.stopPropagation()}>
            <div className="card-head">
              <span className="card-title">{video.title}</span>
              <button className="btn ghost sm" onClick={() => setVideo(null)} aria-label="Close">✕</button>
            </div>
            <video src={video.url} controls autoPlay
                   onLoadedMetadata={(e) => { (e.target as HTMLVideoElement).currentTime = Math.max(0, video.at - 1); }} />
          </div>
        </div>
      )}
    </>
  );
}
