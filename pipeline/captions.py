"""Stage 5a: burned-in captions as an ASS subtitle file.

Style: bold, white text with black outline, centered near the bottom —
similar to the “IN 1915 SHAVING” / “UPPER LIP” Shorts style.
A few words appear at a time, timed to the narration.
"""
from __future__ import annotations

import math
from pathlib import Path


def _ts(seconds: float) -> str:
    """Format seconds as ASS time (H:MM:SS.cc), never rounding up to 60.00."""
    centis = max(0, math.floor(seconds * 100 + 1e-6))
    h, rem = divmod(centis, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _escape(text: str) -> str:
    """Keep narration from being parsed as ASS override tags or line breaks."""
    return (text.replace("\\", "\u29f5").replace("{", "(").replace("}", ")")
            .replace("\r", " ").replace("\n", " "))


def build_ass(scene_audios: list, out_path: Path, cfg: dict) -> None:
    cap = cfg["captions"]
    w, h = cfg["video"]["resolution"]

    # Large, bold, centered captions near the bottom.
    fontsize = int(cap.get("size", 84) * (h / 1080))

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{cap['font']},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,80,80,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    offset = 0.0
    per_line = max(1, int(cap.get("words_per_line", 4)))
    for audio in scene_audios:
        words = audio.words
        if not words:
            offset += audio.duration
            continue
        hold = float(cap.get("hold_seconds", 0.6))
        for i in range(0, len(words), per_line):
            chunk = words[i:i + per_line]
            start = offset + chunk[0][1]
            end = offset + chunk[-1][2]
            # With real timing there are pauses between phrases; keep the caption up
            # through a short pause rather than blinking off and on.
            if i + per_line < len(words):
                end = max(end, min(offset + words[i + per_line][1], end + hold))
            if end <= start:
                end = start + 0.05
            text = _escape(" ".join(str(w[0]) for w in chunk))
            lines.append(f"Dialogue: 0,{_ts(start)},{_ts(end)},Default,,0,0,0,,{text}")
        offset += audio.duration
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
