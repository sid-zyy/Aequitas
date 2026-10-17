import { useEffect, useState } from "react";
import { api, AuditEntry, AuditVerify } from "../api";

// Admin only: who did what, when, from where. Each entry is sealed onto a hash chain; "Verify" recomputes it.

const when = (ms: number) => new Date(ms).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "medium", hour12: false });

function detailText(d: unknown): string {
  if (d == null) return "";
  if (typeof d !== "object") return String(d);
  return Object.entries(d as Record<string, unknown>)
    .map(([k, v]) => `${k.replace(/_/g, " ")}: ${typeof v === "string" && v.length > 24 ? v.slice(0, 12) + "…" : JSON.stringify(v)}`)
    .join(" · ");
}

export default function AuditLog() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [actions, setActions] = useState<string[]>([]);
  const [action, setAction] = useState("");
  const [user, setUser] = useState("");
  const [target, setTarget] = useState("");
  const [chain, setChain] = useState<AuditVerify | null>(null);
  const [checking, setChecking] = useState(false);

  const load = () => api.audit({ action, user, target, limit: 300 }).then((r) => { setEntries(r.entries); setActions(r.actions); });
  const verify = () => { setChecking(true); api.verifyAudit().then(setChain).finally(() => setChecking(false)); };

  useEffect(() => { load(); }, [action, user, target]); // eslint-disable-line
  useEffect(() => { verify(); }, []);

  const users = Array.from(new Set(entries.map((e) => e.username).filter(Boolean))) as string[];

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Audit log</h1>
          <p className="page-sub">Every sign-in, view, decision and export, in order. Each entry is sealed to the one before it, so an edit or a deletion shows.</p>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center" }}>
          {chain && (
            <span className={"pill " + (chain.ok ? "ok" : "alert")} title={chain.head ? `latest hash ${chain.head}` : undefined}>
              <span className={"dot " + (chain.ok ? "live" : "red")} />{" "}
              {chain.ok ? `Chain intact · ${chain.checked} entries verified${chain.ms != null ? ` in ${chain.ms} ms` : ""}`
                : `Chain broken at entry #${chain.broken_at}: ${chain.reason}`}
            </span>
          )}
          <button className="btn" disabled={checking} onClick={verify}>{checking ? "Verifying…" : "Verify chain"}</button>
        </div>
      </div>
      <section className="card">
        <div className="card-head" style={{ gap: 10, flexWrap: "wrap" }}>
          <span className="card-title">Entries</span>
          <span className="card-sub">latest {entries.length}</span>
          <select className="input" value={action} onChange={(e) => setAction(e.target.value)} style={{ marginLeft: "auto", width: "auto" }}>
            <option value="">All actions</option>
            {actions.map((a) => <option key={a} value={a}>{a.replace(/_/g, " ")}</option>)}
          </select>
          <select className="input" value={user} onChange={(e) => setUser(e.target.value)} style={{ width: "auto" }}>
            <option value="">All users</option>
            {users.map((u) => <option key={u} value={u}>{u}</option>)}
          </select>
          <input className="input" placeholder="Target (report, recording, event/…)" value={target}
            onChange={(e) => setTarget(e.target.value)} style={{ width: 240 }} />
        </div>
        {!entries.length && <div className="empty">No entries match.</div>}
        {!!entries.length && (
          <table className="queue">
            <thead>
              <tr><th>#</th><th>When</th><th>Who</th><th>Action</th><th>Target</th><th>Detail</th><th>From</th><th>Seal</th></tr>
            </thead>
            <tbody>
              {entries.map((e) => (
                <tr key={e.id} style={chain && !chain.ok && chain.broken_at === e.id ? { outline: "1px solid var(--danger)" } : undefined}>
                  <td className="mono">{e.id}</td>
                  <td className="mono" style={{ whiteSpace: "nowrap" }}>{when(e.created_at)}</td>
                  <td>{e.username ?? <span className="note">citizen / system</span>}</td>
                  <td>{e.action.replace(/_/g, " ")}</td>
                  <td className="mono">{e.target}</td>
                  <td className="desc" style={{ fontSize: 12 }}>{detailText(e.detail)}</td>
                  <td className="mono note">{e.ip}</td>
                  <td className="mono note" title={e.hash ?? undefined}>{e.hash ? e.hash.slice(0, 8) : "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </main>
  );
}
