import { isIncident, label, LiveCamera, secs, timeOf } from "../api";
import BrowserCamera from "./BrowserCamera";

export default function CameraTile({ cam, running }: { cam: LiveCamera; running: boolean }) {
  const w = cam.latest;
  const alerting = isIncident(w?.label);
  const fresh = w ? Date.now() - w.created_at < 15000 : false;

  return (
    <article className={"card tile" + (alerting ? " alerting" : "")}>
      <div className="tile-head">
        <div style={{ minWidth: 0 }}>
          <div className="tile-name">{cam.name}</div>
          <div className="tile-id">{cam.id.toUpperCase()}</div>
        </div>
        <div className="tile-status">
          {running && cam.has_footage ? (
            <><span className={"dot " + (fresh ? "live" : "busy")} /> {fresh ? "analysing" : "waiting for AI"}</>
          ) : (
            <><span className="dot" /> idle</>
          )}
        </div>
      </div>

      <div className="video-wrap">
        {!cam.has_footage ? (
          <div className="nofeed">No feed</div>
        ) : cam.browser_camera ? (
          <BrowserCamera cameraId={cam.id} running={running} />
        ) : cam.live_source ? (
          <img src={cam.video_url} alt={`${cam.name} live`} />
        ) : (
          <video src={cam.video_url} autoPlay muted loop playsInline />
        )}
        {cam.has_footage && (
          <div className="video-overlay"><span className="rec" /> {cam.live_source ? "LIVE" : "DEMO FEED"}</div>
        )}
        {w && (
          <div className="video-overlay right mono">
            analysed {timeOf(w.created_at)}{cam.behind_live_s != null ? ` · ${cam.behind_live_s.toFixed(1)} s behind live` : ""}
          </div>
        )}
      </div>

      <div className="tile-body">
        <div className="section-label">
          ◆ Scene understanding {cam.view === "close" && <span className="chip">close-range</span>}
        </div>
        {w ? (
          <>
            <div className="timings">
              <span className="chip accent">Summary {secs(w.summarize_ms)}</span>
              <span className="chip accent">Label {secs(w.classify_ms, 2)}</span>
              {w.violence != null && <span className="chip saffron">Clip model {(w.violence * 100).toFixed(0)}%</span>}
              {w.source?.startsWith("clip_model") && <span className="chip saffron">raised by clip model</span>}
              {w.source?.includes("rechecked:") && (
                <span className="chip accent">re-checked by {w.source.split("rechecked:")[1]}</span>
              )}
            </div>
            <p className="summary">{w.summary ?? "—"}</p>
            {alerting ? (
              <div className="alert-box">
                <span>⚠</span>
                <div><b>{label(w.label)}</b>{w.confidence != null && <> · {(w.confidence * 100).toFixed(0)}%</>}
                  <div style={{ fontSize: 12 }}>Needs officer confirmation</div></div>
              </div>
            ) : (
              <div className="normal-box">● {w.label ? "Normal activity" : "AI stage failed for this window"}</div>
            )}
            {cam.recent.length > 1 && (
              <>
                <div className="section-label" style={{ color: "var(--muted)" }}>Running summary</div>
                <ul className="history">
                  {cam.recent.slice(1, 4).map((r) => (
                    <li key={r.created_at}>
                      <span className="t mono">{timeOf(r.created_at)}</span>
                      <span>{isIncident(r.label) ? <b style={{ color: "#ffb3bb" }}>{label(r.label)}: </b> : null}{r.summary}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </>
        ) : (
          <p className="summary" style={{ color: "var(--faint)" }}>
            {cam.has_footage ? (running ? "Waiting for the first analysis…" : "Select the locality to start analysis.") : "No footage for this camera."}
          </p>
        )}
      </div>
    </article>
  );
}
