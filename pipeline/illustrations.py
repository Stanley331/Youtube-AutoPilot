"""Create original, locally rendered motion-comic panels with Pillow.

Style: flat vector cartoons with thick outlines, consistent characters across
scenes in a single video, and bold centered on-screen captions — similar to
the “1915 shaving” / “upper lip” style YouTube Shorts.

FFmpeg adds a gentle camera move (zoom/pan) when assembling the panels.
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .fonts import find_font

# ---------------------------------------------------------------------------
# Palette: one cohesive 4-color scheme per video. top/middle/accent/paper.
# ---------------------------------------------------------------------------

PALETTES = {
    "trench": ((60, 50, 42), (110, 85, 65), (200, 160, 90), (235, 228, 205)),
    "field":  ((110, 140, 70), (160, 190, 110), (235, 210, 130), (245, 240, 220)),
    "night":  ((25, 30, 45), (60, 70, 100), (240, 200, 120), (230, 225, 215)),
    "city":   ((45, 55, 75), (90, 105, 130), (220, 180, 110), (240, 235, 225)),
    "bright": ((50, 85, 120), (120, 160, 185), (245, 200, 110), (248, 245, 230)),
}


def _palette_for(content: str) -> str:
    if any(w in content for w in ("trench", "war", "soldier", "army", "battle", "1915", "wwi", "world war")):
        return "trench"
    if any(w in content for w in ("field", "grass", "hill", "countryside", "farm", "pasture")):
        return "field"
    if any(w in content for w in ("night", "dark", "moon", "star", "lantern", "storm")):
        return "night"
    if any(w in content for w in ("city", "street", "building", "train", "shop", "bridge")):
        return "city"
    return "bright"


# ---------------------------------------------------------------------------
# Deterministic character design (same look every scene, same video)
# ---------------------------------------------------------------------------

class Character:
    """A consistent cartoon character generated once per video."""

    def __init__(self, description: str, width: int, height: int):
        seed = hashlib.sha256(description.lower().encode("utf-8")).digest()
        self.w = width
        self.h = height
        scale = min(width, height)

        # Deterministic but varied appearance from the seed
        s = list(seed)
        self.skin = (240, 190 + s[0] % 30, 150 + s[1] % 40)
        self.hair = (40 + s[2] % 60, 30 + s[3] % 50, 35 + s[4] % 40)
        self.shirt = (70 + s[5] % 80, 90 + s[6] % 70, 110 + s[7] % 60)
        self.pants = (50 + s[8] % 60, 60 + s[9] % 50, 80 + s[10] % 40)
        self.boots = (30, 25, 20)
        self.outline = (30, 25, 20)

        # Body proportions
        self.head_r = int(scale * 0.09)
        self.body_w = int(scale * 0.18)
        self.body_h = int(scale * 0.22)
        self.arm_len = int(scale * 0.16)
        self.leg_len = int(scale * 0.14)

    def draw(self, draw: ImageDraw.Draw, cx: int, cy: int,
             expression: str = "neutral", holding: str | None = None,
             hat: bool = True, uniform: bool = True) -> None:
        s = min(self.w, self.h)
        base_y = int(self.h * 0.82)  # feet position
        head_y = int(self.h * 0.50)

        # Legs
        lx = cx - int(self.body_w * 0.22)
        rx = cx + int(self.body_w * 0.22)
        draw.line((lx, base_y - self.leg_len, lx, base_y), fill=self.outline, width=max(3, s // 180))
        draw.line((rx, base_y - self.leg_len, rx, base_y), fill=self.outline, width=max(3, s // 180))
        draw.line((cx - self.body_w // 2, base_y, cx + self.body_w // 2, base_y),
                  fill=self.outline, width=max(4, s // 140))  # ground contact
        draw.ellipse((lx - 6, base_y - 4, lx + 6, base_y + 4), fill=self.boots)
        draw.ellipse((rx - 6, base_y - 4, rx + 6, base_y + 4), fill=self.boots)

        # Body
        body_top = head_y + self.head_r
        draw.rounded_rectangle((cx - self.body_w // 2, body_top,
                                cx + self.body_w // 2, body_top + self.body_h),
                               radius=int(s * 0.02), fill=self.shirt,
                               outline=self.outline, width=max(3, s // 200))
        # Belt
        draw.line((cx - self.body_w // 2, body_top + int(self.body_h * 0.55),
                   cx + self.body_w // 2, body_top + int(self.body_h * 0.55)),
                  fill=self.outline, width=max(3, s // 160))

        # Arms
        arm_y = body_top + int(self.body_h * 0.35)
        if holding:
            draw.line((cx - self.body_w // 2 - self.arm_len, arm_y,
                       cx - self.body_w // 2 - self.arm_len * 0.7, arm_y - s * 0.04),
                      fill=self.skin, width=max(4, s // 160))
            draw.line((cx + self.body_w // 2 + self.arm_len, arm_y,
                       cx + self.body_w // 2 + self.arm_len * 0.7, arm_y - s * 0.04),
                      fill=self.skin, width=max(4, s // 160))
        else:
            draw.line((cx - self.body_w // 2 - self.arm_len, arm_y,
                       cx - self.body_w // 2 - int(self.arm_len * 0.85), arm_y + s * 0.02),
                      fill=self.skin, width=max(4, s // 160))
            draw.line((cx + self.body_w // 2 + self.arm_len, arm_y,
                       cx + self.body_w // 2 + int(self.arm_len * 0.85), arm_y + s * 0.02),
                      fill=self.skin, width=max(4, s // 160))

        # Head
        draw.ellipse((cx - self.head_r, head_y - self.head_r,
                      cx + self.head_r, head_y + self.head_r),
                     fill=self.skin, outline=self.outline, width=max(3, s // 180))

        # Hair
        draw.pieslice((cx - self.head_r * 1.05, head_y - self.head_r * 1.1,
                       cx + self.head_r * 1.05, head_y + self.head_r * 0.55),
                      180, 360, fill=self.hair)

        # Eyes
        eye_r = max(5, int(self.head_r * 0.09))
        eye_dx = int(self.head_r * 0.38)
        eye_y = head_y - int(self.head_r * 0.02)
        if expression == "angry":
            # Angry brows + narrow eyes
            draw.line((cx - eye_dx - 8, eye_y - 10, cx - eye_dx + 2, eye_y - 16),
                      fill=self.outline, width=max(3, s // 200))
            draw.line((cx + eye_dx + 8, eye_y - 10, cx + eye_dx - 2, eye_y - 16),
                      fill=self.outline, width=max(3, s // 200))
            draw.ellipse((cx - eye_dx - eye_r, eye_y - eye_r,
                          cx - eye_dx + eye_r, eye_y + eye_r), fill=self.outline)
            draw.ellipse((cx + eye_dx - eye_r, eye_y - eye_r,
                          cx + eye_dx + eye_r, eye_y + eye_r), fill=self.outline)
            # Small angry mouth
            draw.arc((cx - int(self.head_r * 0.22), head_y + int(self.head_r * 0.30),
                      cx + int(self.head_r * 0.22), head_y + int(self.head_r * 0.55)),
                     0, 180, fill=self.outline, width=max(4, s // 150))
        elif expression == "happy":
            draw.ellipse((cx - eye_dx - eye_r, eye_y - eye_r,
                          cx - eye_dx + eye_r, eye_y + eye_r), fill=self.outline)
            draw.ellipse((cx + eye_dx - eye_r, eye_y - eye_r,
                          cx + eye_dx + eye_r, eye_y + eye_r), fill=self.outline)
            draw.arc((cx - int(self.head_r * 0.25), head_y + int(self.head_r * 0.35),
                      cx + int(self.head_r * 0.25), head_y + int(self.head_r * 0.60)),
                     0, 180, fill=self.outline, width=max(4, s // 150))
        else:
            draw.ellipse((cx - eye_dx - eye_r, eye_y - eye_r,
                          cx - eye_dx + eye_r, eye_y + eye_r), fill=self.outline)
            draw.ellipse((cx + eye_dx - eye_r, eye_y - eye_r,
                          cx + eye_dx + eye_r, eye_y + eye_r), fill=self.outline)
            draw.line((cx - int(self.head_r * 0.18), head_y + int(self.head_r * 0.38),
                       cx + int(self.head_r * 0.18), head_y + int(self.head_r * 0.38)),
                      fill=self.outline, width=max(3, s // 180))

        # Hat (helmet / cap)
        if hat:
            draw.pieslice((cx - self.head_r * 1.0, head_y - self.head_r * 0.62,
                           cx + self.head_r * 1.0, head_y + self.head_r * 0.42),
                          180, 360, fill=self.shirt, outline=self.outline,
                          width=max(2, s // 220))
            draw.rectangle((cx - self.head_r * 0.92, head_y + self.head_r * 0.30,
                            cx + self.head_r * 0.92, head_y + self.head_r * 0.55),
                           fill=self.shirt, outline=self.outline,
                           width=max(2, s // 220))

        # Optional held object
        if holding:
            self._draw_held(draw, cx, cy, arm_y, holding, s)

    def _draw_held(self, draw: ImageDraw.Draw, cx: int, cy: int, arm_y: int,
                   holding: str, s: int) -> None:
        """Draw a simple held prop (razor, letter, flower, etc.)."""
        o = self.outline
        if holding == "razor":
            draw.rectangle((cx + int(self.body_w * 0.55), arm_y - int(s * 0.02),
                            cx + int(self.body_w * 0.75), arm_y + int(s * 0.02)),
                           fill=(190, 190, 200), outline=o, width=2)
            draw.rectangle((cx + int(self.body_w * 0.75), arm_y - int(s * 0.015),
                            cx + int(self.body_w * 0.95), arm_y + int(s * 0.015)),
                           fill=(120, 120, 130), outline=o, width=2)
        elif holding == "letter":
            draw.rectangle((cx + int(self.body_w * 0.55), arm_y - int(s * 0.035),
                            cx + int(self.body_w * 0.85), arm_y + int(s * 0.035)),
                           fill=(240, 230, 200), outline=o, width=2)
            draw.line((cx + int(self.body_w * 0.70), arm_y - int(s * 0.035),
                       cx + int(self.body_w * 0.70), arm_y + int(s * 0.035)),
                      fill=o, width=2)


# ---------------------------------------------------------------------------
# Backgrounds
# ---------------------------------------------------------------------------

def _draw_trench_background(draw: ImageDraw.Draw, w: int, h: int,
                            palette: tuple, time_of_day: str = "day") -> None:
    top, middle, accent, paper = palette
    sky = top if time_of_day == "day" else (15, 20, 35)
    ground = middle

    # Sky gradient
    for y in range(int(h * 0.30)):
        t = y / (h * 0.30)
        c = tuple(int(sky[i] * (1 - t) + 200 * t) for i in range(3))
        draw.line((0, y, w, y), fill=c)
    # Sun / moon
    cx, cy, r = int(w * 0.8), int(h * 0.18), int(w * 0.07)
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=accent)
    # Distant hills
    draw.ellipse((-w * 0.2, h * 0.15, w * 1.2, h * 0.35), fill=(120, 150, 90))
    # Bare trees on hill
    for tx in (w * 0.12, w * 0.30, w * 0.62, w * 0.88):
        draw.line((tx, h * 0.22, tx, h * 0.30), fill=(40, 35, 30), width=max(2, w // 300))
        draw.line((tx - w * 0.025, h * 0.24, tx + w * 0.025, h * 0.24), fill=(40, 35, 30), width=max(2, w // 300))
        draw.line((tx, h * 0.24, tx, h * 0.27), fill=(40, 35, 30), width=max(2, w // 300))

    # Trench wall (wooden planks)
    wall_top = int(h * 0.36)
    plank_w = w // 8
    for i in range(9):
        x = i * plank_w - plank_w // 4
        draw.rectangle((x, wall_top, x + plank_w, h), fill=ground,
                       outline=(40, 32, 24), width=max(2, w // 250))
        # Wood grain lines
        for gy in range(wall_top + 20, h, 40):
            draw.line((x, gy, x + plank_w, gy), fill=(35, 28, 20), width=max(1, w // 400))

    # Sandbags / parapet top
    bag_top = wall_top - int(h * 0.06)
    for i in range(10):
        bx = i * (w // 10) - w // 20
        draw.ellipse((bx, bag_top, bx + w // 10, bag_top + int(h * 0.07)),
                     fill=(140, 110, 80), outline=(60, 50, 40), width=max(2, w // 250))

    # Ground inside trench
    draw.rectangle((0, wall_top, w, h), fill=(45, 38, 30))


def _draw_field_background(draw: ImageDraw.Draw, w: int, h: int,
                           palette: tuple) -> None:
    top, middle, accent, paper = palette
    # Sky
    for y in range(int(h * 0.35)):
        t = y / (h * 0.35)
        c = tuple(int(top[i] * (1 - t) + 220 * t) for i in range(3))
        draw.line((0, y, w, y), fill=c)
    draw.ellipse((int(w * 0.85), int(h * 0.12), int(w * 0.85) + int(w * 0.09),
                  int(h * 0.12) + int(w * 0.09)), fill=accent)
    # Hills
    draw.ellipse((-w * 0.3, h * 0.20, w * 1.3, h * 0.45), fill=(140, 170, 100))
    draw.ellipse((w * 0.2, h * 0.25, w * 1.1, h * 0.50), fill=(110, 140, 70))
    # Ground
    draw.rectangle((0, int(h * 0.55), w, h), fill=middle)
    # Simple trees
    for tx in (w * 0.08, w * 0.55, w * 0.92):
        trunk_w = w // 40
        draw.rectangle((tx - trunk_w // 2, int(h * 0.62), tx + trunk_w // 2, h),
                       fill=(80, 60, 40))
        draw.ellipse((tx - w * 0.07, int(h * 0.42), tx + w * 0.07, int(h * 0.72)),
                     fill=(90, 130, 60))


def _draw_indoor_background(draw: ImageDraw.Draw, w: int, h: int,
                            palette: tuple) -> None:
    top, middle, accent, paper = palette
    wall = tuple(min(255, c + 25) for c in top)
    # Walls
    draw.rectangle((0, 0, w, int(h * 0.60)), fill=wall)
    draw.rectangle((0, int(h * 0.60), w, h), fill=middle)
    # Floorboard line
    draw.line((0, int(h * 0.60), w, int(h * 0.60)), fill=accent, width=max(2, w // 300))
    # Window
    wx, wy, ww, wh = w * 0.25, h * 0.10, w * 0.20, h * 0.25
    draw.rectangle((wx, wy, wx + ww, wy + wh), fill=(200, 220, 240), outline=accent, width=max(3, w // 200))
    draw.line((wx + ww // 2, wy, wx + ww // 2, wy + wh), fill=accent, width=max(2, w // 300))
    draw.line((wx, wy + wh // 2, wx + ww, wy + wh // 2), fill=accent, width=max(2, w // 300))
    # Desk / table
    draw.rectangle((w * 0.55, h * 0.62, w * 0.95, h * 0.90), fill=(140, 100, 70), outline=top, width=max(3, w // 200))


def draw_background(draw: ImageDraw.Draw, w: int, h: int, content: str,
                    palette_name: str, palette: tuple) -> None:
    if any(wd in content for wd in ("trench", "war", "soldier", "army", "battle", "1915", "world war")):
        _draw_trench_background(draw, w, h, palette)
    elif any(wd in content for wd in ("room", "office", "workshop", "inside", "desk")):
        _draw_indoor_background(draw, w, h, palette)
    else:
        _draw_field_background(draw, w, h, palette)


# ---------------------------------------------------------------------------
# Text (bold centered caption style like the reference)
# ---------------------------------------------------------------------------

def _draw_caption(draw: ImageDraw.Draw, text: str, w: int, h: int,
                  font: ImageFont.FreeTypeFont, y_center: float = 0.86) -> None:
    """Big bold white text with black outline, centered near the bottom."""
    text = re.sub(r"\s+", " ", text).upper().strip()
    if not text:
        return
    # Wrap to fit width
    max_w = w * 0.92
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and draw.textbbox((0, 0), trial, font=font)[2] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lines = lines[:2]  # max two lines

    line_h = draw.textbbox((0, 0), "Ag", font=font)[3]
    total_h = line_h * len(lines) + 14 * (len(lines) - 1)
    y = int(h * y_center) - total_h // 2

    stroke = max(2, w // 350)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (w - tw) // 2
        # Black outline (draw 4 offsets)
        for dx in (-stroke, 0, stroke):
            for dy in (-stroke, 0, stroke):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += line_h + 14


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    path = find_font(bold=bold)
    if path:
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            pass
    return ImageFont.load_default(size=size)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_panel(scene: dict, out_dir: Path, index: int, cfg: dict,
                 script_context: dict | None = None) -> Path:
    """Draw a scene-specific illustrated panel and return its cached PNG path."""
    context = script_context or {}
    key_data = "|".join([
        str(scene.get("narration", "")), str(scene.get("visual_description", "")),
        str(scene.get("setting", "")), str(scene.get("visual_subject", "")),
        str(scene.get("action", "")), "|".join(scene.get("visual_objects", [])),
        str(scene.get("on_screen_text", "")), str(context.get("main_character", "")),
        str(context.get("content_type", "")), str(cfg["video"]["resolution"]),
        "nocaption" if context.get("no_caption") else "",
    ])
    digest = hashlib.sha256(key_data.encode("utf-8")).hexdigest()[:10]
    art_dir = out_dir / "illustrations"
    art_dir.mkdir(parents=True, exist_ok=True)
    path = art_dir / f"scene_{index:02d}_{digest}.png"
    if path.exists():
        return path

    width, height = cfg["video"]["resolution"]
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    content = " ".join(str(scene.get(key, "")) for key in (
        "narration", "visual_query", "visual_description", "setting", "visual_subject", "action"
    )).lower()

    palette_name = _palette_for(content)
    palette = PALETTES[palette_name]
    top, middle, accent, paper = palette

    # --- Background ---
    draw_background(draw, width, height, content, palette_name, palette)

    # --- Consistent character (one per video) ---
    char_desc = str(context.get("main_character", "")) or "soldier"
    char = Character(char_desc, width, height)

    subject = str(scene.get("visual_subject", "")).lower()
    action = str(scene.get("action", "")).lower()
    on_screen = str(scene.get("on_screen_text", "")).strip()

    cx = int(width * 0.50)
    cy = int(height * 0.60)

    # Decide expression & props from action / narration
    expression = "neutral"
    holding = None
    hat = True
    if any(w in action for w in ("shave", "razor", "cut")):
        holding = "razor"
        expression = "neutral"
    elif any(w in action for w in ("smile", "laugh", "happy", "joy")):
        expression = "happy"
    elif any(w in action for w in ("angry", "shout", "mad", "strict")):
        expression = "angry"
    if any(w in action for w in ("letter", "note", "message")):
        holding = "letter"

    # If two people mentioned, draw a second smaller figure
    if " and " in subject or any(w in content for w in ("two", "both", "couple", "pair")):
        char.draw(draw, int(width * 0.38), cy, expression="neutral", hat=hat, uniform=True)
        # Second character: slightly different by re-seeding with a tweak
        char2 = Character(char_desc + " second", width, height)
        char2.draw(draw, int(width * 0.68), cy, expression=expression, hat=hat, uniform=True)
    else:
        char.draw(draw, cx, cy, expression=expression, holding=holding, hat=hat, uniform=True)

    # --- Caption ---
    bold_font = _font(max(44, min(width, height) // 20), bold=True)
    # Burned captions sit at the bottom; in a wide frame the panel's own text goes to the top.
    y_center = 0.14 if width > height else 0.86
    if on_screen:
        _draw_caption(draw, on_screen, width, height, bold_font, y_center=y_center)
    elif not context.get("no_caption"):   # lofi scenes carry no text at all
        fallback = _fallback_title(str(scene.get("visual_query", "")))
        _draw_caption(draw, fallback, width, height, bold_font, y_center=y_center)

    # --- Frame border ---
    margin = int(width * 0.03)
    draw.rounded_rectangle((margin, margin, width - margin, height - margin),
                           radius=int(width * 0.03), outline=accent,
                           width=max(3, width // 220))

    image.save(path, format="PNG", optimize=True)
    return path


def _fallback_title(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text)
    return " ".join(words[:4]) or "A MOMENT TO REMEMBER"
