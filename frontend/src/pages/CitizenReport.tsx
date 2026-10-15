import { FormEvent, useState } from "react";
import { api, ApiError } from "../api";
import { BrandMark, Tricolour } from "../components/Header";

type Done = { report_id: string; tracking_code: string; status: string };

export default function CitizenReport() {
  const [file, setFile] = useState<File | null>(null);
  const [description, setDescription] = useState("");
  const [location, setLocation] = useState("");
  const [when, setWhen] = useState("");
  const [contact, setContact] = useState("");
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<Done | null>(null);
  const [checkId, setCheckId] = useState("");
  const [checkCode, setCheckCode] = useState("");
  const [checkResult, setCheckResult] = useState<string | null>(null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    const f = new FormData();
    f.append("file", file);
    f.append("description", description);
    f.append("location", location);
    if (when) f.append("incident_at", when);
    f.append("contact", contact);
    f.append("consent", String(consent));
    try {
      setDone(await api.submitReport(f));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setBusy(false);
    }
  }

  async function check(e: FormEvent) {
    e.preventDefault();
    try {
      const r = await api.reportStatus(checkId.trim(), checkCode.trim());
      setCheckResult(`${r.report_id}: ${r.status}`);
    } catch {
      setCheckResult("No report found with that ID and tracking code.");
    }
  }

  return (
    <>
      <Tricolour />
      <div className="portal">
        <div className="brand" style={{ marginBottom: 22 }}>
          <BrandMark size={40} />
          <div>
            <div className="brand-name" style={{ fontSize: 18 }}>AEQUITAS · Citizen report</div>
            <div className="brand-sub">Report harassment or an incident to the police · <b>ಸುರಕ್ಷಿತ ಬೆಂಗಳೂರು</b></div>
          </div>
        </div>

        {done ? (
          <section className="card card-body">
            <h1>Thank you. Your report was received.</h1>
            <p className="page-sub">Keep these to check the status. The tracking code is shown only once.</p>
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", margin: "16px 0" }}>
              <div><div className="label">Report ID</div><div className="code-box mono">{done.report_id}</div></div>
              <div><div className="label">Tracking code</div><div className="code-box mono">{done.tracking_code}</div></div>
            </div>
            <p className="note">In an emergency call 112. This portal is not monitored around the clock.</p>
          </section>
        ) : (
          <form className="card card-body" onSubmit={submit}>
            <h1>Report an incident</h1>
            <p className="page-sub" style={{ marginBottom: 18 }}>
              Share a video of harassment or a crime you witnessed. You do not need an account. In an emergency, call <b>112</b>.
            </p>
            <div className="form-row">
              <label className="label">Video (MP4, MOV, AVI… up to 500 MB)</label>
              <input className="input" type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              <p className="note">Please upload the original file from your phone, not a forwarded copy: it helps verification.</p>
            </div>
            <div className="form-row">
              <label className="label">What happened? (at least 10 characters)</label>
              <textarea className="textarea" rows={4} value={description} onChange={(e) => setDescription(e.target.value)}
                placeholder="E.g. A man followed a woman from the bus stop and grabbed her arm." />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="form-row">
                <label className="label">Where? (optional)</label>
                <input className="input" value={location} onChange={(e) => setLocation(e.target.value)} placeholder="e.g. MG Road bus stop" />
              </div>
              <div className="form-row">
                <label className="label">When? (optional)</label>
                <input className="input" type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
              </div>
            </div>
            <div className="form-row">
              <label className="label">Contact (optional, seen only by the police)</label>
              <input className="input" value={contact} onChange={(e) => setContact(e.target.value)} placeholder="Phone or email" />
            </div>
            <label style={{ display: "flex", gap: 8, alignItems: "flex-start", margin: "6px 0 14px" }}>
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span>I agree that this video may be shared with the police for investigation.</span>
            </label>
            {error && <p className="error">{error}</p>}
            <button className="btn primary" disabled={busy || !file || description.trim().length < 10 || !consent}>
              {busy ? "Uploading…" : "Submit report"}
            </button>
          </form>
        )}

        <form className="card card-body" style={{ marginTop: 16 }} onSubmit={check}>
          <div className="card-title" style={{ marginBottom: 10 }}>Check the status of a report</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 10 }}>
            <input className="input" placeholder="Report ID" value={checkId} onChange={(e) => setCheckId(e.target.value)} />
            <input className="input" placeholder="Tracking code" value={checkCode} onChange={(e) => setCheckCode(e.target.value)} />
            <button className="btn navy" disabled={!checkId || !checkCode}>Check</button>
          </div>
          {checkResult && <p style={{ marginBottom: 0 }}>{checkResult}</p>}
        </form>
        <p className="note" style={{ marginTop: 18 }}>Demo system, not an official government service.</p>
      </div>
    </>
  );
}
