// Thin client for the FastAPI backend (/api/*, documented at /api/docs). Cookies carry the session.

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const r = await fetch(path, { credentials: "same-origin", ...init });
  if (r.status === 401 && !path.startsWith("/api/auth/login")) {
    window.dispatchEvent(new Event("aequitas:signed-out"));
  }
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const j = await r.json();
      detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* not JSON */ }
    throw new ApiError(r.status, detail);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export type User = { user_id: number; username: string; role: string };

export type Window = {
  summary: string | null; label: string | null; confidence: number | null;
  summarize_ms: number | null; classify_ms: number | null; lag_ms: number | null;
  violence: number | null; source: string | null; created_at: number; end_ts: number;
};

export type LiveCamera = {
  id: string; name: string; has_footage: boolean; live_source: boolean; view: string; video_url: string; browser_camera?: boolean;
  poster_url?: string | null;
  latest: Window | null; recent: Window[]; behind_live_s: number | null;
};

export type EventItem = {
  id: number; camera_id: string; camera_name: string; area?: string | null; start_ts: number; end_ts: number; category: string;
  status: string; confidence: number; summary: string | null; keyframe_url: string;
  video: { url: string; offset_s: number } | null;
  subtype?: string | null; review?: string | null;
  relevance?: number | null; matched?: string | null;
};

/** An analysed window the chat found by meaning: what a camera saw, incident or not. */
export type Moment = {
  id: number; camera_id: string; camera_name: string; area?: string | null; start_ts: number; end_ts: number;
  summary: string | null; label: string | null; relevance: number | null;
  video: { url: string; offset_s: number } | null;
};

/** Sent back with the next question so follow-ups ("and yesterday?", "the second one") work. */
export type ChatContext = { filter: Record<string, unknown>; event_ids: number[]; mode: string };

export type ChatIndex = {
  enabled: boolean; ready: boolean; engine?: string; mode?: string; embedder?: string; dim?: number;
  events?: number; windows?: number; last_sync_ms?: number | null;
};

export type LiveState = {
  locality: { id: string; name: string }; running: boolean; paused: boolean; error: string | null;
  cameras: LiveCamera[]; concerns: EventItem[];
  model: { backend: string; name: string; concurrency: number };
};

export type Locality = { id: string; name: string; cameras: { id: string; name: string; has_footage: boolean }[] };

export type Report = {
  report_id: string; description: string; location: string | null; incident_at: number | null;
  contact: string | null; sha256: string; duplicate_of: string | null; status: string; notes: string | null;
  created_at: number; updated_at: number; video_url: string;
  language: string; urgent: boolean; translation: Translation | null;
  authenticity: null | {
    ai_likelihood_pct: number | null; verdict: string; reasons: string[]; disclaimer: string; error?: string;
  };
  analysis: { status: string | null; events_found: number | null; total_s: number | null };
  events?: EventItem[];
  audit?: { action: string; created_at: number; username: string | null }[];
};

export type Translation = { text: string; ms: number; model: string; from: string; from_name: string; cached?: boolean };

export type AssistTurn = {
  state: "ask" | "emergency" | "report"; reply: string; urgent: boolean;
  choices: { value: "yes" | "no"; label: string }[]; helplines: { number: string; label: string }[];
};

export type ActiveQueue = {
  reports: (Report & { categories: string[];
    findings: { category: string; status: string; at_s: number; summary: string | null }[] })[];
  counts: { received: number; under_review: number; action_taken: number };
};

/** The citizen portal is a separate public site: open it in its own window, not inside the police app. */
export type StageSummary = {
  jobs: number; ok: number; skipped_static: number; skipped_gate: number;
  summarize_ms: { total: number | null; mean: number | null }; classify_ms: { total: number | null; mean: number | null };
};
export type Recording = {
  id: string; filename: string; officer: string | null; status: string; progress: number; duration_s: number;
  start_ts: number; created_at: number; total_s: number | null; events_found: number; error: string | null;
  vlm_model: string; vlm_backlog: number; stages: StageSummary; view: string | null;
  summary: { overview: string; key_points: { time: string; t: number; point: string }[]; model?: string } | null;
  summary_s: number | null;
};
export type BodyWindow = { t: number; end: number; label: string | null; summary: string | null; source: string | null; violence: number | null };
export type BodyEvent = {
  id: number; t: number; end: number; category: string; status: string; confidence: number; summary: string | null;
  review: "confirmed" | "dismissed" | null; reviewed_by: string | null; keyframe_url: string; headline?: string;
};
export type RecordingDetail = Recording & { windows: BodyWindow[]; events: BodyEvent[]; video_url: string; headline?: string };
export type VideoAnswer = { answer: string; cited_s: number[]; ms: number; model: string };
type History = { q: string; a: string }[];

export type HealthCheck = { name: string; status: "ok" | "warn" | "down"; detail: string };
export type Health = {
  status: "ok" | "warn" | "down"; checked_at: number; uptime_s: number; checks: HealthCheck[];
  vlm: { backend: string; model: string; online: boolean; ping_ms: number | null; loaded: { model: string; vram_mb: number }[] };
  gpu: { name: string; memory_used_mb: number; memory_total_mb: number; utilization_pct: number; temperature_c: number } | null;
  pipeline: { windows: number; failed: number; summarize_ms_p50: number | null; classify_ms_p50: number | null;
    lag_s_p50: number | null; dropped_15min: number; sample_fps: number; mode: string };
  audit: AuditVerify; storage: { free_gb: number; db_mb: number };
};
export type AuditVerify = { ok: boolean; checked: number; broken_at: number | null; reason: string | null; head?: string; head_id?: number | null; ms?: number };
export type AuditEntry = {
  id: number; action: string; target: string | null; detail: unknown; ip: string | null; created_at: number;
  hash: string | null; username: string | null; role: string | null;
};

/** Download URL for an incident's evidence package (a ZIP; the server records the export in the audit log). */
export const evidenceUrl = (eventId: number) => `/api/events/${eventId}/evidence`;

export function openCitizenPortal() {
  const w = window.open("/report", "aequitas-citizen-portal", "popup,width=560,height=900");
  if (w) w.focus();
}

export const api = {
  me: () => request<User>("/api/auth/me"),
  login: (username: string, password: string) => request<User>("/api/auth/login", json({ username, password })),
  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  localities: () => request<Locality[]>("/api/live/localities"),
  startLive: (id: string) => request<LiveState>(`/api/live/${id}/start`, { method: "POST" }),
  liveState: (id: string) => request<LiveState>(`/api/live/${id}`),
  liveAi: () => request<{ paused: boolean }>("/api/live/ai"),
  setLiveAi: (paused: boolean) => request<{ paused: boolean }>("/api/live/ai", json({ paused })),
  reports: () => request<Report[]>("/api/reports"),
  activeReports: () => request<ActiveQueue>("/api/reports/active"),
  report: (id: string) => request<Report>(`/api/reports/${id}`),
  updateReport: (id: string, body: { status?: string; notes?: string }) =>
    request<Report>(`/api/reports/${id}`, { ...json(body), method: "PATCH" }),
  translateReport: (id: string) => request<Translation>(`/api/reports/${id}/translate`, { method: "POST" }),
  assist: (body: { lang: string; message?: string; choice?: "yes" | "no" }) =>
    request<AssistTurn>("/api/citizen/assist", json(body)),
  submitReport: (form: FormData) =>
    request<{ report_id: string; status: string; message: string }>(
      "/api/citizen/reports", { method: "POST", body: form }),
  recordings: () => request<Recording[]>("/api/bodycam"),
  recording: (id: string) => request<RecordingDetail>(`/api/bodycam/${id}`),
  uploadRecording: (form: FormData) => request<Recording>("/api/bodycam", { method: "POST", body: form }),
  summarizeRecording: (id: string) => request<Recording>(`/api/bodycam/${id}/summary`, { method: "POST" }),
  reviewMoment: (id: string, eventId: number, decision: "confirmed" | "dismissed" | null) =>
    request<BodyEvent>(`/api/bodycam/${id}/events/${eventId}/review`, json({ decision })),
  askRecording: (id: string, question: string, history: History) =>
    request<VideoAnswer>(`/api/bodycam/${id}/ask`, json({ question, history })),
  reportAnalysis: (id: string) => request<RecordingDetail>(`/api/reports/${id}/analysis`),
  summarizeReport: (id: string) => request<RecordingDetail>(`/api/reports/${id}/summary`, { method: "POST" }),
  reviewReportMoment: (id: string, eventId: number, decision: "confirmed" | "dismissed" | null) =>
    request<BodyEvent>(`/api/reports/${id}/events/${eventId}/review`, json({ decision })),
  askReport: (id: string, question: string, history: History) =>
    request<VideoAnswer>(`/api/reports/${id}/ask`, json({ question, history })),
  health: () => request<Health>("/api/health"),
  audit: (q: { action?: string; user?: string; target?: string; before_id?: number; limit?: number } = {}) => {
    const p = new URLSearchParams(Object.entries(q).filter(([, v]) => v != null && v !== "").map(([k, v]) => [k, String(v)]));
    return request<{ entries: AuditEntry[]; actions: string[] }>(`/api/audit?${p}`);
  },
  verifyAudit: () => request<AuditVerify>("/api/audit/verify"),
  reportStatus: (id: string) =>
    request<{ report_id: string; status: string; submitted_at: number; updated_at: number }>(
      `/api/citizen/reports/${encodeURIComponent(id)}`),
};

export type StreamEvent =
  | {
      type: "retrieved"; total: number; head: string[]; tail: string[]; events: EventItem[]; moments: Moment[];
      mode: "filter" | "hybrid" | "detail" | "similar"; kept: string[]; weak: boolean; earlier: boolean; suggestions: string[]; context: ChatContext;
      intent: string; parsed_with: string | null; warnings: string[];
      parse_ms: number; query_ms: number | null; embed_ms: number | null; vector_ms: number | null; retrieve_ms: number;
    }
  | { type: "token"; text: string }
  | { type: "summary"; summary: string | null }
  | { type: "done"; elapsed_ms: number; summary_ttft_ms: number | null; summary_ms: number | null; parse_ms: number };

export const chatIndex = () => request<ChatIndex>("/api/chat/index");

/** Streams the grounded chatbot answer (server-sent events over a POST). */
export async function askStream(question: string, onEvent: (e: StreamEvent) => void,
                                context?: ChatContext | null, signal?: AbortSignal): Promise<void> {
  const r = await fetch("/api/ask/stream", { ...json({ question, context: context ?? null }), credentials: "same-origin", signal });
  if (!r.ok || !r.body) throw new ApiError(r.status, r.statusText);
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const data = block.split("\n").find((l) => l.startsWith("data: "));
      if (data) onEvent(JSON.parse(data.slice(6)));
    }
  }
}

export const label = (c: string | null | undefined) => (c ?? "").replace(/_/g, " ");
export const isIncident = (c: string | null | undefined) => !!c && c !== "none";
export const secs = (ms: number | null | undefined, d = 1) => (ms == null ? "–" : `${(ms / 1000).toFixed(d)} s`);
export const timeOf = (ms: number) =>
  new Date(ms).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
