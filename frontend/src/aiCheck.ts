import { Report } from "./api";

type Authenticity = NonNullable<Report["authenticity"]> & { basis?: string; headline?: string };

/** What the officer screens show for the "AI-generated?" check. A percentage only when real evidence backs it
 * (C2PA declares AI, AI-tool metadata, phone camera metadata). The frame detector alone gives uncalibrated
 * scores (it gives real stock video 100 %), so a frame-only result is shown as words, never as a percentage. */
export function aiCheck(a: Authenticity): { pct: number | null; text: string; color: string; title: string; frameOnly: boolean } {
  // results stored before the basis field existed: the frame detector spoke alone when it is the first reason
  const frameOnly = a.basis ? a.basis === "frames" : /^Frame detector/.test(a.reasons?.[0] ?? "");
  if (frameOnly) {
    const flagged = !!a.basis && a.verdict === "uncertain";
    return {
      // not flagged reads "Real" (prefer a missed fake to a real video flagged); older results
      // stored "Frame check: inconclusive" as their headline, so the title is set here, not taken from them
      pct: null, text: flagged ? "AI-like frames ?" : "Real", color: flagged ? "var(--warn)" : "var(--ok)",
      title: flagged ? (a.headline ?? "Frame check: AI-like frames, needs a human look")
        : "Frame check: no AI signs found, treated as real",
      frameOnly,
    };
  }
  const color = a.verdict === "likely_ai" ? "var(--danger)" : a.verdict === "uncertain" ? "var(--warn)" : "var(--ok)";
  if (a.ai_likelihood_pct == null) return { pct: null, text: "no signal", color: "var(--muted)", title: "not enough signal", frameOnly };
  return { pct: a.ai_likelihood_pct, text: `${a.ai_likelihood_pct}%`, color, title: a.verdict.replace(/_/g, " "), frameOnly };
}
