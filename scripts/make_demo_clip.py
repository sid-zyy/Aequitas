"""Builds the citizen-upload demo clip, data/footage/demo/flyover-attack.mp4 (33.7 s, about 7 MB).

Two free Pexels clips from the same shoot (same flyover, same actors), joined: the first 12 s of 10400886 (a man
walks up to a group by a car: nothing happens) and all of 10400819 (two of them fight). On the timeline this
gives about 12 s of "no incident" before the fight, so the demo shows both. Staged stock footage, not a real
incident; Pexels License, attribution not required. Needs about 25 MB of downloads.

    .venv/Scripts/python scripts/make_demo_clip.py"""

import subprocess
import urllib.request
from pathlib import Path

import imageio_ffmpeg

SRC = Path("data/footage/staged_candidates")
OUT = Path("data/footage/demo/flyover-attack.mp4")
CLIPS = {  # Pexels video id -> direct 720p file
    "10400886": "https://videos.pexels.com/video-files/10400886/10400886-hd_1366_720_25fps.mp4",
    "10400819": "https://videos.pexels.com/video-files/10400819/10400819-hd_1366_720_25fps.mp4",
}


def main() -> None:
    SRC.mkdir(parents=True, exist_ok=True)
    for vid, url in CLIPS.items():
        dest = SRC / f"px-{vid}.mp4"
        if not dest.exists():
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r:
                dest.write_bytes(r.read())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    graph = ("[0:v]trim=0:12,setpts=PTS-STARTPTS,scale=1280:-2,fps=25[a];"
             "[1:v]setpts=PTS-STARTPTS,scale=1280:-2,fps=25[b];[a][b]concat=n=2:v=1:a=0[v]")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
                    "-i", str(SRC / "px-10400886.mp4"), "-i", str(SRC / "px-10400819.mp4"),
                    "-filter_complex", graph, "-map", "[v]", "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT)], check=True)
    print(f"ok {OUT} ({OUT.stat().st_size // 2**20} MB)")


if __name__ == "__main__":
    main()
