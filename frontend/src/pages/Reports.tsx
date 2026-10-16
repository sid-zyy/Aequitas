import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, label, openCitizenPortal, Report, timeOf } from "../api";
import { aiCheck } from "../aiCheck";
import ReportText from "../components/ReportText";
import { LANG_NAME } from "../i18n";

const STATUSES = ["received", "under_review", "action_taken", "closed", "rejected"];

function AiMeter({ a }: { a: NonNullable<Report["authenticity"]> }) {
  const ai = aiCheck(a);
  if (ai.pct == null) return <b style={{ color: ai.color }}>{ai.title}</b>;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div className="meter" style={{ width: 120 }}><i style={{ width: `${ai.pct}%`, background: ai.color }} /></div>
      <b className="mono" style={{ color: ai.color }}>{ai.text}</b>
      <span className="badge status">{a.verdict.replace(/_/g, " ")}</span>
    </div>
  );
}

const day = (ms: number) => new Date(ms).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" });

export default function Reports() {
  const [list, setList] = useState<Report[]>([]);
  const [params] = useSearchParams();
  const [sel, setSel] = useState<string | null>(params.get("id"));
  const [detail, setDetail] = useState<Report | null>(null);
  const [notes, setNotes] = useState("");

  const load = () => api.reports().then((r) => { setList(r); if (!sel && r[0]) setSel(r[0].report_id); });
  useEffect(() => { load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, []); // eslint-disable-line
  useEffect(() => {
    if (!sel) return;
    api.report(sel).then((d) => { setDetail(d); setNotes(d.notes ?? ""); });
  }, [sel]);

  async function update(body: { status?: string; notes?: string }) {
    if (!sel) return;
    const d = await api.updateReport(sel, body);
    setDetail(d);
    load();
  }

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">All citizen reports</h1>
          <p className="page-sub">Videos sent by the public. Each is fingerprinted on arrival, analysed by AI, and checked for AI generation.</p>
        </div>
        <button className="btn" style={{ marginLeft: "auto" }} onClick={openCitizenPortal}>Open citizen portal ↗</button>
      </div>
      <div className="split">
        <section className="card">
          <div className="card-head"><span className="card-title">Inbox</span><span className="card-sub">{list.length} reports</span></div>
          {!list.length && <div className="empty">No reports yet. Citizens report at /report.</div>}
          {list.map((r) => (
            <div key={r.report_id} className={"report-row" + (sel === r.report_id ? " sel" : "")} onClick={() => setSel(r.report_id)}>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <b className="mono">{r.report_id}</b>
                {r.urgent && <span className="badge incident" title="The citizen told the portal assistant they need help now">urgent</span>}
                <span className="badge status">{label(r.status)}</span>
                {r.language !== "en" && <span className="badge status">{LANG_NAME[r.language] ?? r.language}</span>}
                {r.authenticity?.verdict === "likely_ai" && <span className="badge incident">likely AI</span>}
                {r.duplicate_of && <span className="badge unverified">duplicate</span>}
                <span className="note" style={{ marginLeft: "auto" }}>{day(r.created_at)}</span>
              </div>
              <div lang={r.language} style={{ color: "#cfdbe3", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{r.description}</div>
            </div>
          ))}
        </section>

        {detail ? (
          <section className="card">
            <div className="card-head">
              <span className="card-title">Report {detail.report_id}</span>
              {detail.urgent && <span className="badge incident">urgent</span>}
              <span className="card-sub">{day(detail.created_at)}</span>
              <select className="select" style={{ width: 180, marginLeft: "auto" }} value={detail.status}
                onChange={(e) => update({ status: e.target.value })}>
                {STATUSES.map((s) => <option key={s} value={s}>{label(s)}</option>)}
              </select>
            </div>
            <div className="card-body" style={{ display: "grid", gap: 16 }}>
              <video className="player" src={detail.video_url} controls />
              <dl className="kv">
                <dt>Citizen says</dt><dd><ReportText reportId={detail.report_id} text={detail.description}
                  language={detail.language} translation={detail.translation} /></dd>
                {detail.urgent && <><dt>Urgent</dt><dd>The citizen said they need help now and was shown 112 and the other helpline numbers.</dd></>}
                <dt>Location</dt><dd>{detail.location ?? "—"}</dd>
                <dt>When</dt><dd>{detail.incident_at ? day(detail.incident_at) : "—"}</dd>
                <dt>Contact</dt><dd>{detail.contact ?? "—"}</dd>
                <dt>Fingerprint</dt><dd className="mono" style={{ fontSize: 11.5 }}>SHA-256 {detail.sha256}</dd>
                {detail.duplicate_of && <><dt>Duplicate of</dt><dd>{detail.duplicate_of}</dd></>}
              </dl>

              <div className="card" style={{ boxShadow: "none" }}>
                <div className="card-head"><span className="card-title">AI-likelihood</span>
                  <span className="card-sub">is this video AI-generated?</span></div>
                <div className="card-body">
                  {!detail.authenticity ? <span className="note">Checking…</span> : detail.authenticity.error ? (
                    <span className="note">Check failed: {detail.authenticity.error}</span>
                  ) : (
                    <>
                      <AiMeter a={detail.authenticity} />
                      <ul style={{ margin: "8px 0", paddingLeft: 18, color: "#cfdbe3" }}>
                        {detail.authenticity.reasons
                          // frame-only: the headline is shown above; results stored before v2 carry the raw score
                          .slice(aiCheck(detail.authenticity).frameOnly ? 1 : 0)
                          .map((r) => <li key={r}>{r}</li>)}
                      </ul>
                      <p className="note">{detail.authenticity.disclaimer}</p>
                    </>
                  )}
                </div>
              </div>

              <div className="card" style={{ boxShadow: "none" }}>
                <div className="card-head"><span className="card-title">AI analysis of the video</span>
                  <span className="card-sub">{detail.analysis.status ?? "—"}{detail.analysis.total_s != null && ` · ${detail.analysis.total_s} s total`}</span></div>
                <div className="card-body" style={{ display: "grid", gap: 8 }}>
                  {!detail.events?.length && <span className="note">No incidents found (or still analysing).</span>}
                  {detail.events?.map((ev) => (
                    <div key={ev.id} style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
                      <span className="badge incident">{label(ev.category)}</span>
                      <span className="mono note">{timeOf(ev.start_ts)}</span>
                      <span>{ev.summary}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <label className="label">Police notes (never shown to the citizen)</label>
                <textarea className="textarea" rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} />
                <button className="btn navy" style={{ marginTop: 8 }} onClick={() => update({ notes })}>Save notes</button>
              </div>

              {!!detail.audit?.length && (
                <div>
                  <div className="label">Audit trail</div>
                  <ul className="history">
                    {detail.audit.map((a, i) => (
                      <li key={i}><span className="t mono">{day(a.created_at)}</span><span>{label(a.action)}{a.username ? ` · ${a.username}` : " · citizen/system"}</span></li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </section>
        ) : <section className="card empty">Select a report.</section>}
      </div>
    </main>
  );
}
