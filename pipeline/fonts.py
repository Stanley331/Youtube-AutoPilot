"""Find a usable system font on macOS, Windows, or Linux."""
import os
import sys
from pathlib import Path

_WINDOWS_FONTS = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"

BOLD = [
    _WINDOWS_FONTS / "arialbd.ttf", _WINDOWS_FONTS / "segoeuib.ttf", _WINDOWS_FONTS / "calibrib.ttf",
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/System/Library/Fonts/Supplemental/Impact.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
]
REGULAR = [
    _WINDOWS_FONTS / "arial.ttf", _WINDOWS_FONTS / "segoeui.ttf",
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def find_font(bold: bool = True, preferred: str = "") -> Path | None:
    candidates = ([Path(preferred)] if preferred else []) + (BOLD if bold else REGULAR)
    return next((p for p in candidates if p.exists()), None)


def ffmpeg_path(p: Path) -> str:
    """Escape a path for use inside an ffmpeg filter string (drive colons, backslashes)."""
    return str(p).replace("\\", "/").replace(":", r"\:")


def is_windows() -> bool:
    return sys.platform.startswith("win")
