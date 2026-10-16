import { Fragment, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ActiveQueue, api, label, openCitizenPortal, RecordingDetail } from "../api";
import { aiCheck } from "../aiCheck";
import ReportText from "../components/ReportText";
import VideoReview, { running } from "../components/VideoReview";

type Row = ActiveQueue["reports"][number];

/** A citizen report's video reviewed like a bodycam recording: summary, timeline, key moments, per-video chat. */
function ReportReview({ r }: { r: Row }) {
  const [d, setD] = useState<RecordingDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let stop = false;
    const tick = () => api.reportAnalysis(r.report_id).then((x) => { if (!stop) setD(x); return x; })
      .catch((e) => { if (!stop) setError((e as Error).message); return null; });
    tick();
    const t = setInterval(() => tick().then((x) => { if (x && !running(x.status)) clearInterval(t); }), 3000);
    return () => { stop = true; clearInterval(t); };
  }, [r.report_id]);

  if (error) return <div className="error">{error}</div>;
  if (!d) return <div className="empty">Loading the analysis…</div>;
  return (
    <VideoReview d={d} setD={setD} api={{
      review: (eventId, decision) => api.reviewReportMoment(r.report_id, eventId, decision),
      summarize: () => api.summarizeReport(r.report_id),
      ask: (q, history) => api.askReport(r.report_id, q, history),
    }} extra={r.authenticity && !r.authenticity.error && (
      <p className="note">AI-generated check: {aiCheck(r.authenticity).frameOnly ? aiCheck(r.authenticity).title : r.authenticity.reasons?.[0]}</p>
    )} />
  );
}

const NEXT: Record<string, { to: string; text: string }[]> = {
  received: [{ to: "under_review", text: "Start review" }, { to: "rejected", text: "Reject" }],
  under_review: [{ to: "action_taken", text: "Action taken" }, { to: "rejected", text: "Reject" }],
  action_taken: [{ to: "closed", text: "Close" }],
};

function age(ms: number) {
  const m = Math.max(0, Math.round((Date.now() - ms) / 60000));
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  return h < 48 ? `${h} h ${m % 60} min` : `${Math.floor(h / 24)} days`;
}

export default function ActiveRequests() {
  const [q, setQ] = useState<ActiveQueue | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [open, setOpen] = useState<string | null>(null);
  const seen = useRef<Set<string> | null>(null);
  const [fresh, setFresh] = useState<Set<string>>(new Set());
  const nav = useNavigate();

  const load = () => api.activeReports().then((d) => {
    const ids = new Set(d.reports.map((r) => r.report_id));
    if (seen.current) setFresh(new Set([...ids].filter((i) => !seen.current!.has(i))));
    seen.current = ids;
    setQ(d);
  });
  useEffect(() => { load(); const t = setInterval(load, 3000); return () => clearInterval(t); }, []); // eslint-disable-line

  async function move(id: string, status: string) {
    await api.updateReport(id, { status });
    load();
  }

  const rows = (q?.reports ?? []).filter((r) => filter === "all" || r.status === filter);
  const total = q ? q.reports.length : 0;

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Active requests</h1>
          <p className="page-sub">Citizen reports still open: urgent first, then longest waiting. Updates every 3 s. Every action is written to the audit trail.</p>
        </div>
        <button className="btn" style={{ marginLeft: "auto" }} onClick={openCitizenPortal}>Open citizen portal ↗</button>
      </div>

      <div className="stat-strip">
        {([["all", "Open", total], ["received", "New, not yet reviewed", q?.counts.received ?? 0],
          ["under_review", "Under review", q?.counts.under_review ?? 0],
          ["action_taken", "Action taken, to close", q?.counts.action_taken ?? 0]] as const).map(([k, t, n]) => (
          <button key={k} className={"card stat stat-btn" + (filter === k ? " sel" : "")} onClick={() => setFilter(k)}>
            <div className="stat-value mono">{n}</div><div className="stat-label">{t}</div>
          </button>
        ))}
      </div>

      <section className="card">
        <div className="card-head">
          <span className="card-title">Queue</span>
          <span className="card-sub">{rows.length} shown</span>
        </div>
        {!q && <div className="empty">Loading…</div>}
        {q && !rows.length && <div className="empty">No open requests. New citizen reports appear here within seconds.</div>}
        {!!rows.length && (
          <div style={{ overflowX: "auto" }}>
            <table className="queue">
              <thead>
                <tr><th>Report</th><th>Waiting</th><th>Status</th><th>What the citizen says</th><th>AI found</th><th>AI-generated?</th><th /></tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const ai = r.authenticity && !r.authenticity.error ? aiCheck(r.authenticity) : null;
                  return (
                    <Fragment key={r.report_id}>
                    <tr className={(fresh.has(r.report_id) ? "fresh" : "") + (open === r.report_id ? " expanded" : "")}>
                      <td><b className="mono">{r.report_id}</b>{r.urgent && <div><span className="badge incident">urgent</span></div>}{r.duplicate_of && <div><span className="badge unverified">duplicate</span></div>}</td>
                      <td className="mono">{age(r.created_at)}</td>
                      <td><span className="badge status">{label(r.status)}</span></td>
                      <td className="desc"><ReportText reportId={r.report_id} text={r.description} language={r.language}
                        translation={r.translation} />{r.location && <div className="note">{r.location}</div>}</td>
                      <td>
                        {r.analysis.status && r.analysis.status !== "done" && <span className="note">{r.analysis.status}…</span>}
                        {r.categories.map((c) => <span key={c} className="badge incident" style={{ marginRight: 4 }}>{label(c)}</span>)}
                        {r.analysis.status === "done" && !r.categories.length && <span className="badge none">nothing flagged</span>}
                      </td>
                      <td className="mono">
                        {!r.authenticity ? <span className="note">checking…</span> : !ai ? <span className="note">check failed</span>
                          : ai.pct == null ? <span style={{ color: ai.color }} title={ai.title}>{ai.text}</span> : (
                          <b style={{ color: ai.color }} title={ai.title}>{ai.text}{r.authenticity.verdict === "uncertain" && " ?"}</b>
                        )}
                      </td>
                      <td style={{ whiteSpace: "nowrap" }}>
                        {(NEXT[r.status] ?? []).map((n) => (
                          <button key={n.to} className={"btn" + (n.to === "rejected" ? " ghost" : " navy")} style={{ marginRight: 6 }}
                            onClick={() => move(r.report_id, n.to)}>{n.text}</button>
                        ))}
                        <button className="btn" style={{ marginRight: 6 }}
                          onClick={() => setOpen(open === r.report_id ? null : r.report_id)}>
                          {open === r.report_id ? "▲ Hide" : "▶ Analyse"}
                        </button>
                        <button className="btn ghost" onClick={() => nav(`/reports?id=${r.report_id}`)}>Open</button>
                      </td>
                    </tr>
                    {open === r.report_id && (
                      <tr className="review-row">
                        <td colSpan={7}>
                          <ReportReview r={r} />
                        </td>
                      </tr>
                    )}
                    </Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
