"""Stage 6: a 1280x720 thumbnail from a video frame with big outlined text.

Style: bold yellow/white text with thick black outline, centered —
designed to stand out in the YouTube Shorts feed.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from .fonts import find_font
from .util import duration_of, run


def make_thumbnail(video: Path, text: str, out_path: Path, cfg: dict, at_seconds: float = 3.0) -> Path:
    frame = out_path.with_name("frame.jpg")
    # Never seek past the end of a very short video, which would yield no frame.
    try:
        total = duration_of(video)
        at_seconds = max(0.0, min(at_seconds, total - 0.2))
    except Exception:
        at_seconds = 0.0
    run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{at_seconds:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(frame)])
    if not frame.exists():
        raise RuntimeError(f"could not extract a frame from {video}")
    # Keep standard 16:9 thumbnail dimensions without stretching portrait frames.
    img = ImageOps.fit(Image.open(frame).convert("RGB"), (1280, 720), method=Image.Resampling.LANCZOS)
    img = ImageEnhance.Brightness(img).enhance(0.50)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    draw = ImageDraw.Draw(img)

    font_path = find_font(bold=True, preferred=cfg["thumbnail"].get("font_path", ""))
    words = (text or "").upper().split()
    lines = [" ".join(words[:2]), " ".join(words[2:])] if len(words) > 2 else [" ".join(words)]
    lines = [l for l in lines if l]
    size = 160 if len(lines) == 1 else 130
    font = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)
    # Shrink the font until the widest line fits inside the frame.
    while size > 40 and lines and max(draw.textbbox((0, 0), l, font=font)[2] for l in lines) > 1200:
        size -= 10
        font = ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)

    total_h = sum(draw.textbbox((0, 0), l, font=font)[3] for l in lines) + 30 * max(0, len(lines) - 1)
    y = (720 - total_h) // 2
    stroke = max(8, 1280 // 120)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (1280 - tw) // 2
        # Thick black outline for readability over any background
        for dx in (-stroke, 0, stroke):
            for dy in (-stroke, 0, stroke):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill="#FFD400",
                  stroke_width=max(2, stroke // 3), stroke_fill=(0, 0, 0))
        y += bbox[3] - bbox[1] + 30
    img.save(out_path, quality=92)
    frame.unlink(missing_ok=True)
    return out_path
