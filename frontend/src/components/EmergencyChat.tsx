import { FormEvent, useEffect, useRef, useState } from "react";
import { api, ApiError, AssistTurn } from "../api";
import { Lang, STRINGS } from "../i18n";

type Msg = { who: "bot" | "me"; text: string; turn?: AssistTurn };
type Step = null | "video" | "what" | "where" | "when" | "contact" | "consent" | "review";
type Draft = { file: File | null; description: string; location: string; when: string; contact: string };
export type Submitted = { report_id: string; status: string };

const EMPTY: Draft = { file: null, description: "", location: "", when: "", contact: "" };
const VIDEO_EXT = /\.(mp4|mov|avi|mkv|webm|m4v)$/i;

/** datetime-local value for now, in local time. */
function nowLocal() {
  const d = new Date();
  d.setMinutes(d.getMinutes() - d.getTimezoneOffset());
  return d.toISOString().slice(0, 16);
}

/** The portal's assistant. First it asks whether the person needs help now: those replies come from the server's
 *  fixed, rule-based text (aequitas/emergency.py), so no number is ever made up and nothing claims help is sent.
 *  Then it can take the report itself: attach the video with 📎 and answer a few fixed questions; the report goes
 *  through the same endpoint as the form. */
export default function EmergencyChat({ lang, urgent, onUrgent, onSubmitted }: {
  lang: Lang; urgent: boolean; onUrgent: (urgent: boolean) => void; onSubmitted: (r: Submitted) => void;
}) {
  const t = STRINGS[lang];
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState(false); // a popup in the bottom-right corner; the conversation is kept while closed
  const [step, setStep] = useState<Step>(null);
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [when, setWhen] = useState(nowLocal());
  const end = useRef<HTMLDivElement>(null);
  const picker = useRef<HTMLInputElement>(null);

  const bot = (text: string) => setMsgs((m) => [...m, { who: "bot", text }]);
  const me = (text: string) => setMsgs((m) => [...m, { who: "me", text }]);

  async function turn(body: { message?: string; choice?: "yes" | "no" }, shown?: string) {
    setBusy(true);
    try {
      const r = await api.assist({ lang, ...body });
      setMsgs((m) => [...m, ...(shown ? [{ who: "me" as const, text: shown }] : []), { who: "bot", text: r.reply, turn: r }]);
      if (r.state === "emergency") onUrgent(true);
    } finally {
      setBusy(false);
    }
  }

  function restart() {
    setMsgs([]);
    setStep(null);
    setDraft(EMPTY);
    turn({});
  }

  // Start (and restart in the new language when it changes).
  useEffect(() => { restart(); }, [lang]); // eslint-disable-line

  useEffect(() => { if (open) end.current?.scrollIntoView({ block: "nearest" }); }, [msgs, open, step]);

  function startReport() {
    setStep("video");
    bot(t.askVideo);
  }

  function attach(f: File | undefined) {
    if (picker.current) picker.current.value = "";
    if (!f) return;
    if (!f.type.startsWith("video/") && !VIDEO_EXT.test(f.name)) {
      bot(t.notVideo);
      return;
    }
    me(`📎 ${f.name} · ${(f.size / 1024 ** 2).toFixed(1)} MB`);
    setDraft((d) => ({ ...d, file: f }));
    if (step === null || step === "video") {
      setStep("what");
      bot(t.gotVideo.replace("{name}", f.name));
    }
  }

  function answer(value: string, shown?: string) {
    if (shown ?? value) me(shown ?? value);
    if (step === "what") {
      if (value.trim().length < 10) return bot(t.tooShort);
      setDraft((d) => ({ ...d, description: value.trim() }));
      setStep("where");
      bot(t.askWhere);
    } else if (step === "where") {
      setDraft((d) => ({ ...d, location: value.trim() }));
      setStep("when");
      setWhen(nowLocal());
      bot(t.askWhen);
    } else if (step === "when") {
      setDraft((d) => ({ ...d, when: value }));
      setStep("contact");
      bot(t.askContact);
    } else if (step === "contact") {
      setDraft((d) => ({ ...d, contact: value.trim() }));
      setStep("consent");
      bot(t.askConsent);
    }
  }

  function consent(yes: boolean) {
    me(yes ? t.consentYes : t.consentNo);
    if (!yes) return bot(t.needConsent);
    setStep("review");
    bot(t.reviewIntro.replace("{submit}", t.submit));
  }

  async function submit() {
    if (!draft.file) return;
    setBusy(true);
    const f = new FormData();
    f.append("file", draft.file);
    f.append("description", draft.description);
    f.append("location", draft.location);
    if (draft.when) f.append("incident_at", draft.when);
    f.append("contact", draft.contact);
    f.append("consent", "true");
    f.append("language", lang);
    f.append("urgent", String(urgent));
    try {
      onSubmitted(await api.submitReport(f));
    } catch (err) {
      bot(err instanceof ApiError ? err.message : t.uploadFailed);
    } finally {
      setBusy(false);
    }
  }

  function send(e: FormEvent) {
    e.preventDefault();
    const m = text.trim();
    if (!m) return;
    setText("");
    if (step === "what" || step === "where" || step === "contact") answer(m);
    else turn({ message: m }, m);
  }

  const fmtWhen = (v: string) => (v ? new Date(v).toLocaleString(lang === "en" ? "en-IN" : `${lang}-IN`,
    { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }) : t.notGiven);
  const last = msgs[msgs.length - 1]?.turn;
  const typing = step === null || step === "what" || step === "where" || step === "contact";
  return (
    <>
    {!open && (
      <button className="chat-fab" lang={lang} onClick={() => setOpen(true)} aria-label={t.chatTitle}>
        <span aria-hidden>💬</span> {t.chatTitle}
      </button>
    )}
    <section className="card chat-card chat-popup" lang={lang} aria-live="polite" hidden={!open}>
      <div className="card-head">
        <span className="card-title">{t.chatTitle}</span>
        <span className="card-sub">{t.chatSub}</span>
        <button className="btn ghost" style={{ marginLeft: "auto", padding: "4px 10px" }}
          onClick={() => { onUrgent(false); restart(); }}>{t.restart}</button>
        <button className="btn ghost chat-close" onClick={() => setOpen(false)} aria-label="Close">×</button>
      </div>
      <div className="card-body chat-body">
        {msgs.map((m, i) => (
          <div key={i} className={"chat-msg " + m.who}>
            <div className="chat-bubble">{m.text}</div>
            {!!m.turn?.helplines.length && (
              <div className="helplines">
                <div className="label">{t.numbers}</div>
                <ul>
                  {m.turn.helplines.map((h) => (
                    <li key={h.number}>
                      <span className="helpline-number mono">{h.number}</span>
                      <span>{h.label}</span>
                      <a className="btn ghost" href={`tel:${h.number}`} aria-label={`${h.label} ${h.number}`}>📞</a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        {step === null && !!last?.choices.length && (
          <div className="chat-choices">
            {last.choices.map((c) => (
              <button key={c.value} className={"btn " + (c.value === "yes" ? "danger" : "navy")} disabled={busy}
                onClick={() => turn({ choice: c.value }, c.label)}>{c.label}</button>
            ))}
          </div>
        )}
        {step === null && (last?.state === "report" || last?.state === "emergency") && (
          <div className="chat-choices">
            <button className="btn primary" onClick={startReport}>📎 {t.reportInChat}</button>
          </div>
        )}
        {step === "where" && (
          <div className="chat-choices"><button className="btn ghost" onClick={() => answer("", t.skip)}>{t.skip}</button></div>
        )}
        {step === "when" && (
          <div className="chat-choices" style={{ flexWrap: "wrap" }}>
            <button className="btn navy" onClick={() => answer(nowLocal(), t.justNow)}>{t.justNow}</button>
            <input className="input" type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)}
              style={{ width: "auto" }} aria-label={t.askWhen} />
            <button className="btn navy" disabled={!when} onClick={() => answer(when, fmtWhen(when))}>{t.useTime}</button>
            <button className="btn ghost" onClick={() => answer("", t.skip)}>{t.skip}</button>
          </div>
        )}
        {step === "contact" && (
          <div className="chat-choices"><button className="btn ghost" onClick={() => answer("", t.skip)}>{t.skip}</button></div>
        )}
        {step === "consent" && (
          <div className="chat-choices">
            <button className="btn primary" onClick={() => consent(true)}>{t.consentYes}</button>
            <button className="btn ghost" onClick={() => consent(false)}>{t.consentNo}</button>
          </div>
        )}
        {step === "review" && draft.file && (
          <div className="chat-review">
            {urgent && <span className="badge incident">{t.urgent}</span>}
            <dl>
              <dt>{t.rvVideo}</dt><dd>{draft.file.name}</dd>
              <dt>{t.rvWhat}</dt><dd>{draft.description}</dd>
              <dt>{t.rvWhere}</dt><dd>{draft.location || t.notGiven}</dd>
              <dt>{t.rvWhen}</dt><dd>{fmtWhen(draft.when)}</dd>
              <dt>{t.rvContact}</dt><dd>{draft.contact || t.notGiven}</dd>
            </dl>
            <button className="btn primary" disabled={busy} onClick={submit}>{busy ? t.uploading : t.submit}</button>
          </div>
        )}
        <div ref={end} />
      </div>
      <form className="chat-input" onSubmit={send}>
        <input ref={picker} type="file" accept="video/*" hidden onChange={(e) => attach(e.target.files?.[0])} />
        <button type="button" className="btn ghost chat-attach" title={t.attach} aria-label={t.attach}
          disabled={busy || step === "review"} onClick={() => picker.current?.click()}>📎</button>
        <input className="input" lang={lang} value={text} onChange={(e) => setText(e.target.value)}
          placeholder={t.chatPlaceholder} maxLength={step === "what" ? 2000 : 500} disabled={!typing} />
        <button className="btn navy" disabled={busy || !text.trim() || !typing}>{t.send}</button>
      </form>
    </section>
    </>
  );
}
