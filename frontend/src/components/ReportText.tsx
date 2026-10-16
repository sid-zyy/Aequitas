import { MouseEvent, useEffect, useState } from "react";
import { api, ApiError, Translation } from "../api";
import { LANG_NAME } from "../i18n";

/** What the citizen wrote, exactly as written, plus (for Hindi/Kannada) a button that asks the local LLM for an
 *  English translation shown underneath. The translation is timed and cached on the report. */
export default function ReportText({ reportId, text, language, translation }: {
  reportId: string; text: string; language: string; translation: Translation | null;
}) {
  const [tr, setTr] = useState<Translation | null>(translation);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { setTr(translation); setErr(null); }, [reportId, translation?.text]); // eslint-disable-line

  async function translate(e: MouseEvent) {
    e.stopPropagation();
    setBusy(true);
    setErr(null);
    try {
      setTr(await api.translateReport(reportId));
    } catch (x) {
      setErr(x instanceof ApiError ? x.message : "Translation failed");
    } finally {
      setBusy(false);
    }
  }

  const foreign = language !== "en";
  return (
    <div>
      <div lang={language}>{text}</div>
      {foreign && (
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4, flexWrap: "wrap" }}>
          <span className="badge status">{LANG_NAME[language] ?? language}</span>
          {!tr && (
            <button className="btn" style={{ padding: "3px 10px" }} disabled={busy} onClick={translate}>
              {busy ? "Translating…" : "Translate to English"}
            </button>
          )}
          {err && <span className="error" style={{ margin: 0 }}>{err}</span>}
        </div>
      )}
      {tr && (
        <div className="translation" lang="en">
          <div>{tr.text}</div>
          <div className="note">
            English translation by {tr.model} · {(tr.ms / 1000).toFixed(1)} s{tr.cached !== false ? " (saved earlier)" : ""} · machine translation, check with the original
          </div>
        </div>
      )}
    </div>
  );
}
