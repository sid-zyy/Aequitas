import { Fragment, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ActiveQueue, api, label, openCitizenPortal } from "../api";

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
          <p className="page-sub">Citizen reports still open, longest waiting first. Updates every 3 s. Every action is written to the audit trail.</p>
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
                  const pct = r.authenticity?.ai_likelihood_pct;
                  return (
                    <Fragment key={r.report_id}>
                    <tr className={(fresh.has(r.report_id) ? "fresh" : "") + (open === r.report_id ? " expanded" : "")}>
                      <td><b className="mono">{r.report_id}</b>{r.duplicate_of && <div><span className="badge unverified">duplicate</span></div>}</td>
                      <td className="mono">{age(r.created_at)}</td>
                      <td><span className="badge status">{label(r.status)}</span></td>
                      <td className="desc">{r.description}{r.location && <div className="note">{r.location}</div>}</td>
                      <td>
                        {r.analysis.status && r.analysis.status !== "done" && <span className="note">{r.analysis.status}…</span>}
                        {r.categories.map((c) => <span key={c} className="badge incident" style={{ marginRight: 4 }}>{label(c)}</span>)}
                        {r.analysis.status === "done" && !r.categories.length && <span className="badge none">nothing flagged</span>}
                      </td>
                      <td className="mono">
                        {!r.authenticity ? <span className="note">checking…</span> : pct == null ? <span className="note">no signal</span> : (
                          <b style={{ color: pct >= 85 ? "var(--danger)" : pct >= 50 ? "var(--warn)" : "var(--ok)" }}>{pct}%</b>
                        )}
                      </td>
                      <td style={{ whiteSpace: "nowrap" }}>
                        {(NEXT[r.status] ?? []).map((n) => (
                          <button key={n.to} className={"btn" + (n.to === "rejected" ? " ghost" : " navy")} style={{ marginRight: 6 }}
                            onClick={() => move(r.report_id, n.to)}>{n.text}</button>
                        ))}
                        <button className="btn" style={{ marginRight: 6 }}
                          onClick={() => setOpen(open === r.report_id ? null : r.report_id)}>
                          {open === r.report_id ? "▲ Hide" : "▶ Watch"}
                        </button>
                        <button className="btn ghost" onClick={() => nav(`/reports?id=${r.report_id}`)}>Open</button>
                      </td>
                    </tr>
                    {open === r.report_id && (
                      <tr className="review-row">
                        <td colSpan={7}>
                          <div className="review">
                            <video className="player" src={r.video_url} controls autoPlay muted playsInline />
                            <div>
                              <div className="section-label">What the AI saw (needs officer confirmation)</div>
                              {!r.findings.length && <p className="note">Nothing flagged{r.analysis.status !== "done" ? " yet: still analysing" : ""}.</p>}
                              <ul className="history">
                                {r.findings.map((f, i) => (
                                  <li key={i}>
                                    <span className="t mono">{f.at_s >= 0 && f.at_s < 36000 ? `${f.at_s.toFixed(0)} s` : ""}</span>
                                    <span><b style={{ color: "#ffb3bb" }}>{label(f.category)}</b>
                                      {f.status !== "confirmed" && <span className="note"> (single window)</span>}: {f.summary}</span>
                                  </li>
                                ))}
                              </ul>
                              {r.authenticity && !r.authenticity.error && (
                                <p className="note">AI-generated check: {r.authenticity.reasons?.[0]}</p>
                              )}
                            </div>
                          </div>
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
