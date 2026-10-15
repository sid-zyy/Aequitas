import { useEffect, useRef, useState } from "react";

// The page's own camera (source: browser in config/bangalore.yaml). The browser asks for permission; frames are
// sent to this laptop's server at the pipeline's sample rate while the locality is being analysed.
const FPS = 4;
const MAX_W = 640;

export default function BrowserCamera({ cameraId, running }: { cameraId: string; running: boolean }) {
  const video = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(0);

  async function ask() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser blocks the camera on this address. Open the app at http://localhost:8000.");
      return;
    }
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      setStream(s);
    } catch (e) {
      const name = (e as DOMException).name;
      setError(name === "NotAllowedError"
        ? "Camera permission was refused. Allow it from the camera icon in the address bar, then press Allow camera."
        : name === "NotFoundError" ? "No camera found on this device."
        : name === "NotReadableError" ? "The camera is in use by another app (or by the server: set source: browser)."
        : `Camera error: ${(e as Error).message}`);
    }
  }

  useEffect(() => { ask(); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (video.current && stream) video.current.srcObject = stream;
    return () => stream?.getTracks().forEach((t) => t.stop());
  }, [stream]);

  useEffect(() => {
    if (!stream || !running) return;
    const canvas = document.createElement("canvas");
    let busy = false;
    const t = setInterval(() => {
      const v = video.current;
      if (busy || !v || v.readyState < 2 || !v.videoWidth) return;
      const scale = Math.min(1, MAX_W / v.videoWidth);
      canvas.width = Math.round(v.videoWidth * scale);
      canvas.height = Math.round(v.videoHeight * scale);
      canvas.getContext("2d")?.drawImage(v, 0, 0, canvas.width, canvas.height);
      busy = true;
      canvas.toBlob((blob) => {
        if (!blob) { busy = false; return; }
        fetch(`/api/live/cameras/${cameraId}/frame`, {
          method: "POST", body: blob, credentials: "same-origin", headers: { "Content-Type": "image/jpeg" },
        }).then((r) => r.ok && setSent((n) => n + 1)).catch(() => undefined).finally(() => { busy = false; });
      }, "image/jpeg", 0.8);
    }, 1000 / FPS);
    return () => clearInterval(t);
  }, [stream, running, cameraId]);

  if (!stream) {
    return (
      <div className="nofeed" style={{ flexDirection: "column", gap: 10, padding: 16, textAlign: "center" }}>
        <div>{error ?? "Waiting for camera permission…"}</div>
        <button className="btn primary" onClick={ask}>Allow camera</button>
        <div className="note">Frames go only to this laptop's Aequitas server. No face recognition.</div>
      </div>
    );
  }
  return (
    <>
      <video ref={video} autoPlay muted playsInline style={{ transform: "scaleX(-1)" }} />
      <div className="video-overlay right mono" style={{ top: 8, bottom: "auto" }}>
        {running ? `sending ${FPS} fps · ${sent} frames` : "select this locality to start analysis"}
      </div>
    </>
  );
}
