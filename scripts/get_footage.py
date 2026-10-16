"""Downloads the demo footage listed in config/footage.txt (free Pexels clips, 720p) into data/footage/.
About 320 MB in total. Already-downloaded files are skipped.

    .venv/Scripts/python scripts/get_footage.py"""

import time
import urllib.request
from pathlib import Path

LIST = Path("config/footage.txt")
OUT = Path("data/footage")


def main() -> None:
    for line in LIST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        rel, vid, *_ = line.split()
        vid, _, width = vid.partition("@")  # "7120813@1366": a clip whose 720p file is not 1280 wide
        dest = OUT / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f"https://www.pexels.com/download/video/{vid}/?h=720&w={width or 1280}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, dest.open("wb") as f:
            f.write(r.read())
        print(f"ok {rel} ({dest.stat().st_size // 2**20} MB)")
        time.sleep(2)  # be polite to the server


if __name__ == "__main__":
    main()
