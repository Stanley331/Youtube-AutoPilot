"""Small shared helpers: shell, ffprobe, slugs, logging."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # A Windows console without UTF-8 (the Streamlit process, say) must not sink a title with an emoji.
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="replace").decode(encoding, errors="replace"), flush=True)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command, raising with the captured stderr on failure."""
    # Decode as UTF-8 with replacement so non-ASCII output from ffmpeg or
    # PowerShell never raises UnicodeDecodeError on Windows code pages.
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        sys.stderr.write((result.stderr or "")[-4000:])
        raise RuntimeError(f"command failed: {' '.join(str(c) for c in cmd[:3])} ...")


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    ).stdout.strip()
    return float(out)


def slugify(text: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "video"


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
