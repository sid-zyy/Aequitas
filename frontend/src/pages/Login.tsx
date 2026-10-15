import { FormEvent, useState } from "react";
import { api, ApiError, User } from "../api";
import { BrandMark, Tricolour } from "../components/Header";

export default function Login({ onSignedIn }: { onSignedIn: (u: User) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      onSignedIn(await api.login(username, password));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not reach the server");
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Tricolour />
      <div className="login-wrap">
        <section className="login-hero">
          <div className="brand">
            <BrandMark size={40} />
            <div>
              <div className="brand-name" style={{ fontSize: 18 }}>AEQUITAS</div>
              <div className="brand-sub">Public Safety Command · Bengaluru</div>
            </div>
          </div>
          <div>
            <h1>Every camera, understood<br />in seconds.</h1>
            <p>
              Live CCTV, officer bodycam and citizen reports, summarised and classified by AI that runs entirely on
              this machine. Every alert is confirmed by an officer.
            </p>
            <div className="hero-points">
              <div>● Summary → classification → grounded answers, each stage timed</div>
              <div>● No face recognition. Peaceful gatherings are never incidents.</div>
              <div>● Open-source models, served locally: no data leaves the station</div>
            </div>
          </div>
          <div className="note">ಸುರಕ್ಷಿತ ಬೆಂಗಳೂರು · Safe Bengaluru · Demo system, not an official government service</div>
        </section>
        <form className="card login-card" onSubmit={submit}>
          <h2>Officer sign-in</h2>
          <p className="page-sub" style={{ marginBottom: 18 }}>Authorised police personnel only. Access is logged.</p>
          <div className="form-row">
            <label className="label" htmlFor="u">Username</label>
            <input id="u" className="input" autoComplete="username" value={username}
              onChange={(e) => setUsername(e.target.value)} autoFocus />
          </div>
          <div className="form-row">
            <label className="label" htmlFor="p">Password</label>
            <input id="p" className="input" type="password" autoComplete="current-password" value={password}
              onChange={(e) => setPassword(e.target.value)} />
          </div>
          {error && <p className="error">{error}</p>}
          <button className="btn primary" style={{ width: "100%", marginTop: 6 }} disabled={busy || !username || !password}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
          <p className="note" style={{ marginTop: 16 }}>
            Citizen? <a href="/report">Report an incident</a> without an account.
          </p>
        </form>
      </div>
    </>
  );
}
