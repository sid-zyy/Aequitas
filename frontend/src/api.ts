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
  id: string; name: string; has_footage: boolean; live_source: boolean; view: string; video_url: string;
  latest: Window | null; recent: Window[]; behind_live_s: number | null;
};

export type EventItem = {
  id: number; camera_id: string; camera_name: string; start_ts: number; end_ts: number; category: string;
  status: string; confidence: number; summary: string | null; keyframe_url: string;
  video: { url: string; offset_s: number } | null;
};

export type LiveState = {
  locality: { id: string; name: string }; running: boolean; error: string | null;
  cameras: LiveCamera[]; concerns: EventItem[];
  model: { backend: string; name: string; concurrency: number };
};

export type Locality = { id: string; name: string; cameras: { id: string; name: string; has_footage: boolean }[] };

export type Report = {
  report_id: string; description: string; location: string | null; incident_at: number | null;
  contact: string | null; sha256: string; duplicate_of: string | null; status: string; notes: string | null;
  created_at: number; updated_at: number; video_url: string;
  authenticity: null | {
    ai_likelihood_pct: number | null; verdict: string; reasons: string[]; disclaimer: string; error?: string;
  };
  analysis: { status: string | null; events_found: number | null; total_s: number | null };
  events?: EventItem[];
  audit?: { action: string; created_at: number; username: string | null }[];
};

export const api = {
  me: () => request<User>("/api/auth/me"),
  login: (username: string, password: string) => request<User>("/api/auth/login", json({ username, password })),
  logout: () => request<{ ok: boolean }>("/api/auth/logout", { method: "POST" }),
  localities: () => request<Locality[]>("/api/live/localities"),
  startLive: (id: string) => request<LiveState>(`/api/live/${id}/start`, { method: "POST" }),
  liveState: (id: string) => request<LiveState>(`/api/live/${id}`),
  reports: () => request<Report[]>("/api/reports"),
  report: (id: string) => request<Report>(`/api/reports/${id}`),
  updateReport: (id: string, body: { status?: string; notes?: string }) =>
    request<Report>(`/api/reports/${id}`, { ...json(body), method: "PATCH" }),
  submitReport: (form: FormData) =>
    request<{ report_id: string; tracking_code: string; status: string; message: string }>(
      "/api/citizen/reports", { method: "POST", body: form }),
  reportStatus: (id: string, code: string) =>
    request<{ report_id: string; status: string; submitted_at: number; updated_at: number }>(
      `/api/citizen/reports/${encodeURIComponent(id)}?code=${encodeURIComponent(code)}`),
};

export type StreamEvent =
  | { type: "retrieved"; total: number; head: string[]; tail: string[]; events: EventItem[]; parse_ms: number; query_ms: number }
  | { type: "token"; text: string }
  | { type: "summary"; summary: string | null }
  | { type: "done"; elapsed_ms: number; summary_ttft_ms: number | null; summary_ms: number | null; parse_ms: number };

/** Streams the grounded chatbot answer (server-sent events over a POST). */
export async function askStream(question: string, onEvent: (e: StreamEvent) => void): Promise<void> {
  const r = await fetch("/api/ask/stream", { ...json({ question }), credentials: "same-origin" });
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
