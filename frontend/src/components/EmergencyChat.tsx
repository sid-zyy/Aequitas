import { FormEvent, useEffect, useRef, useState } from "react";
import { api, AssistTurn } from "../api";
import { Lang, STRINGS } from "../i18n";

type Msg = { who: "bot" | "me"; text: string; turn?: AssistTurn };

/** The portal's assistant: asks whether the person needs help now. Replies come from the server's fixed,
 *  rule-based text (aequitas/emergency.py), so no number is ever made up and nothing claims help is sent. */
export default function EmergencyChat({ lang, onUrgent }: { lang: Lang; onUrgent: (urgent: boolean) => void }) {
  const t = STRINGS[lang];
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);

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

  // Start (and restart in the new language when it changes).
  useEffect(() => { setMsgs([]); turn({}); }, [lang]); // eslint-disable-line

  useEffect(() => { end.current?.scrollIntoView({ block: "nearest" }); }, [msgs]);

  function send(e: FormEvent) {
    e.preventDefault();
    const m = text.trim();
    if (!m) return;
    setText("");
    turn({ message: m }, m);
  }

  const last = msgs[msgs.length - 1]?.turn;
  return (
    <section className="card chat-card" lang={lang} aria-live="polite">
      <div className="card-head">
        <span className="card-title">{t.chatTitle}</span>
        <span className="card-sub">{t.chatSub}</span>
        <button className="btn ghost" style={{ marginLeft: "auto", padding: "4px 10px" }}
          onClick={() => { setMsgs([]); onUrgent(false); turn({}); }}>{t.restart}</button>
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
        {!!last?.choices.length && (
          <div className="chat-choices">
            {last.choices.map((c) => (
              <button key={c.value} className={"btn " + (c.value === "yes" ? "danger" : "navy")} disabled={busy}
                onClick={() => turn({ choice: c.value }, c.label)}>{c.label}</button>
            ))}
          </div>
        )}
        <div ref={end} />
      </div>
      <form className="chat-input" onSubmit={send}>
        <input className="input" lang={lang} value={text} onChange={(e) => setText(e.target.value)}
          placeholder={t.chatPlaceholder} maxLength={500} />
        <button className="btn navy" disabled={busy || !text.trim()}>{t.send}</button>
      </form>
    </section>
  );
}
