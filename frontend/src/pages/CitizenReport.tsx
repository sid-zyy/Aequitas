import { FormEvent, useEffect, useState } from "react";
import { api, ApiError } from "../api";
import EmergencyChat from "../components/EmergencyChat";
import { BrandMark, Tricolour } from "../components/Header";
import { Lang, LANGS, STRINGS } from "../i18n";

function initialLang(): Lang {
  try {
    const v = localStorage.getItem("aeq-portal-lang");
    if (v === "hi" || v === "kn" || v === "en") return v;
  } catch { /* storage blocked */ }
  return "en";
}

type Done = { report_id: string; tracking_code: string; status: string };

export default function CitizenReport() {
  const [lang, setLang] = useState<Lang>(initialLang);
  const t = STRINGS[lang];
  const [urgent, setUrgent] = useState(false);
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

  useEffect(() => {
    document.documentElement.lang = lang;
    try { localStorage.setItem("aeq-portal-lang", lang); } catch { /* storage blocked */ }
  }, [lang]);

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
    f.append("language", lang);
    f.append("urgent", String(urgent));
    try {
      setDone(await api.submitReport(f));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t.uploadFailed);
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
      setCheckResult(t.notFound);
    }
  }

  const second = lang === "en" ? STRINGS.kn : STRINGS.en; // masthead shows the name in two languages
  return (
    <div className="gov" lang={lang}>
      <a className="skip" href="#main">{t.navReport}</a>
      <Tricolour />
      <div className="gov-util">
        <div className="gov-wrap">
          <span>{t.demo}</span>
          <div className="lang-switch" role="group" aria-label={t.language}>
            {LANGS.map((l) => (
              <button key={l.id} type="button" lang={l.id} className={"lang-btn" + (lang === l.id ? " on" : "")}
                aria-pressed={lang === l.id} onClick={() => setLang(l.id)}>{l.name}</button>
            ))}
          </div>
        </div>
      </div>
      <header className="gov-mast">
        <div className="gov-wrap">
          <BrandMark size={72} />
          <div className="gov-names">
            <div className="gov-title">{t.site}</div>
            <div className="gov-title-2" lang={lang === "en" ? "kn" : "en"}>{second.site}</div>
            <div className="gov-org">{t.org}</div>
          </div>
          <div className="gov-emergency"><span className="gov-112 mono">112</span><span>{t.emergencyBar}</span></div>
        </div>
      </header>
      <nav className="gov-nav" aria-label={t.site}>
        <div className="gov-wrap">
          <a href="#main">{t.navReport}</a>
          <a href="#track">{t.navTrack}</a>
        </div>
      </nav>
      <main className="portal" id="main">
        {!done && <EmergencyChat lang={lang} onUrgent={setUrgent} />}

        {done ? (
          <section className="card card-body">
            <h1>{t.thanks}</h1>
            <p className="page-sub">{t.keep}</p>
            <div style={{ display: "flex", gap: 18, flexWrap: "wrap", margin: "16px 0" }}>
              <div><div className="label">{t.reportId}</div><div className="code-box mono">{done.report_id}</div></div>
              <div><div className="label">{t.trackingCode}</div><div className="code-box mono">{done.tracking_code}</div></div>
            </div>
            <p className="note">{t.notMonitored}</p>
          </section>
        ) : (
          <form className="card card-body" onSubmit={submit}>
            <h1>{t.title}</h1>
            <p className="page-sub" style={{ marginBottom: 18 }}>
              {t.intro} <b>112</b>.
            </p>
            {urgent && (
              <div className="urgent-banner">
                <span className="badge incident">{t.urgent}</span>
                <span>{t.urgentOn}</span>
                <button type="button" className="btn ghost" style={{ marginLeft: "auto", padding: "4px 10px" }}
                  onClick={() => setUrgent(false)}>{t.urgentOff}</button>
              </div>
            )}
            <div className="form-row">
              <label className="label">{t.video}</label>
              <input className="input" type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
              <p className="note">{t.videoNote}</p>
            </div>
            <div className="form-row">
              <label className="label">{t.what}</label>
              <textarea className="textarea" rows={4} lang={lang} value={description} maxLength={2000}
                onChange={(e) => setDescription(e.target.value)} placeholder={t.whatPlaceholder} />
              <p className="note">{t.whatHint}</p>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div className="form-row">
                <label className="label">{t.where}</label>
                <input className="input" lang={lang} value={location} onChange={(e) => setLocation(e.target.value)} placeholder={t.wherePlaceholder} />
              </div>
              <div className="form-row">
                <label className="label">{t.when}</label>
                <input className="input" type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
              </div>
            </div>
            <div className="form-row">
              <label className="label">{t.contact}</label>
              <input className="input" value={contact} onChange={(e) => setContact(e.target.value)} placeholder={t.contactPlaceholder} />
            </div>
            <label style={{ display: "flex", gap: 8, alignItems: "flex-start", margin: "6px 0 14px" }}>
              <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span>{t.consent}</span>
            </label>
            {error && <p className="error">{error}</p>}
            <button className="btn primary" disabled={busy || !file || description.trim().length < 10 || !consent}>
              {busy ? t.uploading : t.submit}
            </button>
          </form>
        )}

        <form className="card card-body" id="track" style={{ marginTop: 16 }} onSubmit={check}>
          <div className="card-title" style={{ marginBottom: 10 }}>{t.checkTitle}</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 10 }}>
            <input className="input" placeholder={t.reportId} value={checkId} onChange={(e) => setCheckId(e.target.value)} />
            <input className="input" placeholder={t.trackingCode} value={checkCode} onChange={(e) => setCheckCode(e.target.value)} />
            <button className="btn navy" disabled={!checkId || !checkCode}>{t.check}</button>
          </div>
          {checkResult && <p style={{ marginBottom: 0 }}>{checkResult}</p>}
        </form>
      </main>
      <footer className="gov-foot">
        <div className="gov-wrap">
          <div className="gov-foot-numbers mono">{t.footerNumbers}</div>
          <div>{t.org} · {t.demo}</div>
        </div>
      </footer>
    </div>
  );
}
