"""Evidence package: one incident as a self-checking ZIP an officer can hand on (case file, court, another unit).

Contents:
- `keyframe.jpg`: the frame grid the AI looked at
- `video/<file>`: the original video, byte for byte (not re-encoded), if it is under the size cap
- `manifest.json`: what the AI said, what the officer decided, when, with which models, the stage timings, the
  source file's SHA-256 (and, for a citizen report, whether it still matches the hash taken on arrival), the
  audit entries for this incident, and the audit chain's state at export time
- `SHA256SUMS`: a hash per file, checkable with `sha256sum -c SHA256SUMS`
- `README.txt`: the same in plain words, and a draft certificate for the responsible officer to complete

The AI's output is marked as a draft throughout: the package records the human decision separately."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

from aequitas import __version__
from aequitas.reports import file_sha256
from aequitas.taxonomy import label

FORMAT = "aequitas-evidence/1"


def _iso(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000).astimezone().isoformat(timespec="seconds") if ms else None


def _source_kind(camera_id: str) -> str:
    if camera_id.startswith("report-"):
        return "citizen_report"
    if camera_id.startswith("bodycam-"):
        return "bodycam"
    return "cctv"


def _audit_entries(auth, targets: list[str]) -> list[dict]:
    seen, out = set(), []
    for t in targets:
        for e in auth.audit_trail(t):
            key = (e["created_at"], e["action"], t)
            if key in seen:
                continue
            seen.add(key)
            out.append({"at": _iso(e["created_at"]), "action": e["action"], "target": t, "by": e["username"],
                        "detail": json.loads(e["detail"]) if e["detail"] else None})
    return sorted(out, key=lambda e: e["at"] or "")


def build_package(store, auth, event_id: int, exported_by: str | None, max_video_mb: int = 512) -> tuple[bytes, dict]:
    """Returns (zip bytes, manifest). Raises LookupError if the event doesn't exist."""
    ev = store.event(event_id)
    if ev is None:
        raise LookupError(event_id)
    upload = store.upload_for_event(ev["camera_id"], ev["start_ts"])
    kind = _source_kind(ev["camera_id"])
    report = None
    if kind == "citizen_report":
        rows = auth._q("SELECT report_id, status, sha256, language, created_at FROM reports WHERE camera_id=?",
                       (ev["camera_id"],))
        report = dict(rows[0]) if rows else None

    files: dict[str, bytes | Path] = {}
    kf = Path(ev["keyframe_path"])
    if kf.is_file():
        files["keyframe.jpg"] = kf

    source = None
    if upload:
        path = Path(upload["path"])
        exists = path.is_file()
        size = path.stat().st_size if exists else None
        included = exists and size <= max_video_mb * 1024**2
        source = {
            "upload_id": upload["upload_id"], "original_filename": upload["filename"], "bytes": size,
            "sha256": file_sha256(path) if exists else None,
            "recorded_start": _iso(upload["start_ts"]), "duration_s": (upload["duration_ms"] or 0) / 1000,
            "incident_offset_s": round(max(0, ev["start_ts"] - upload["start_ts"]) / 1000, 1),
            "included": included,
            "note": None if included else ("file missing" if not exists else f"larger than {max_video_mb} MB: "
                                           "not included; match it against the sha256 above"),
        }
        if included:
            files[f"video/{upload['upload_id']}{path.suffix.lower()}"] = path
        if report:
            source["hash_on_arrival"] = report["sha256"]
            source["unchanged_since_arrival"] = source["sha256"] == report["sha256"]

    targets = [f"event/{event_id}"]
    if upload:
        targets += [upload["upload_id"], f"{upload['upload_id']}/{event_id}"]
    if report:
        targets += [report["report_id"], f"{report['report_id']}/{event_id}"]

    manifest = {
        "package": {"format": FORMAT, "generator": f"Aequitas {__version__}",
                    "exported_at": _iso(int(datetime.now().timestamp() * 1000)), "exported_by": exported_by},
        "incident": {
            "event_id": ev["event_id"], "event_uid": ev["event_uid"], "camera_id": ev["camera_id"],
            "camera_name": ev.get("camera_name"), "source_kind": kind,
            "start": _iso(ev["start_ts"]), "end": _iso(ev["end_ts"]),
            "ai_label": ev["category"], "ai_label_text": label(ev["category"]), "ai_status": ev["status"],
            "ai_confidence": ev["confidence"], "vlm_label": ev.get("vlm_category"), "ai_summary": ev.get("summary"),
            "models": ev.get("model_versions"),
        },
        "human_review": {"decision": ev.get("review"), "by": ev.get("reviewed_by"), "at": _iso(ev.get("reviewed_at"))},
        "stage_timings_ms": {k: v for k, v in (store.event_timings(event_id) or {}).items() if k != "event_id"},
        "source": source,
        "citizen_report": {k: report[k] for k in ("report_id", "status", "language")} if report else None,
        "audit": {"entries": _audit_entries(auth, targets), "chain_at_export": auth.verify_chain()},
    }

    blobs = {name: (p.read_bytes() if isinstance(p, Path) else p) for name, p in files.items()}
    manifest["files"] = {name: hashlib.sha256(b).hexdigest() for name, b in blobs.items()}
    blobs["manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False).encode()
    blobs["README.txt"] = _readme(manifest).encode()
    sums = "".join(f"{hashlib.sha256(b).hexdigest()}  {n}\n" for n, b in blobs.items())
    blobs["SHA256SUMS"] = sums.encode()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, b in blobs.items():
            # video is already compressed; text compresses well
            z.writestr(name, b, zipfile.ZIP_STORED if name.startswith("video/") else zipfile.ZIP_DEFLATED)
    return buf.getvalue(), manifest


def _readme(m: dict) -> str:
    inc, rev, src, chain = m["incident"], m["human_review"], m["source"], m["audit"]["chain_at_export"]
    decision = (f"{rev['decision'].upper()} by {rev['by']} at {rev['at']}" if rev["decision"]
                else "NOT YET REVIEWED by an officer")
    lines = [
        "AEQUITAS EVIDENCE PACKAGE",
        "=" * 25,
        f"Incident #{inc['event_id']} ({inc['event_uid']}) · camera: {inc['camera_name'] or inc['camera_id']}",
        f"When: {inc['start']} to {inc['end']}",
        f"Exported: {m['package']['exported_at']} by {m['package']['exported_by'] or 'unknown'}",
        "",
        "WHAT THE AI SAID (a draft for a human, not a finding)",
        f"  Label: {inc['ai_label_text']} · AI status: {inc['ai_status']} · confidence {inc['ai_confidence']}",
        f"  Summary: {inc['ai_summary'] or '(none)'}",
        "",
        "WHAT THE OFFICER DECIDED",
        f"  {decision}",
        "",
        "SOURCE VIDEO",
    ]
    if src:
        lines += [f"  Original file: {src['original_filename']} ({src['bytes']} bytes), recorded from {src['recorded_start']}",
                  f"  SHA-256: {src['sha256']}",
                  f"  Incident starts {src['incident_offset_s']} s into the file."]
        if "unchanged_since_arrival" in src:
            lines.append("  Matches the hash taken when the citizen uploaded it." if src["unchanged_since_arrival"]
                         else "  WARNING: does NOT match the hash taken when the citizen uploaded it.")
        lines.append("  Included in this package, byte for byte." if src["included"] else f"  Not included: {src['note']}.")
    else:
        lines.append("  Live camera: no stored video file for this moment; the key frame is included.")
    lines += [
        "",
        "AUDIT TRAIL",
        f"  {len(m['audit']['entries'])} entries for this incident (manifest.json).",
        (f"  Audit chain intact at export: {chain['checked']} entries verified, latest hash {chain['head']}."
         if chain["ok"] else f"  WARNING: audit chain broken at entry {chain['broken_at']}: {chain['reason']}."),
        "",
        "CHECKING THIS PACKAGE",
        "  sha256sum -c SHA256SUMS      (Linux/macOS; on Windows: Get-FileHash <file> -Algorithm SHA256)",
        "",
        "DRAFT CERTIFICATE (for the responsible officer to check, complete and sign; not legal advice)",
        "  Under Section 63 of the Bharatiya Sakshya Adhiniyam, 2023 (electronic records), I certify that the",
        "  electronic record(s) listed in SHA256SUMS were produced by the computer system described below, in its",
        "  ordinary use, and that the hash values above identify them.",
        f"  System: Aequitas {m['package']['generator'].split()[-1]}, a local, offline analysis system. Models: "
        f"{json.dumps(inc['models'])}",
        "  Name / designation: ____________________   Signature: ______________   Date: __________",
        "",
    ]
    return "\n".join(lines)
