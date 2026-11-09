"""Doodle explainers: a hand-drawn cartoon on a white page for every sentence of narration.

The format behind channels like "what did ancient humans do all day": a calm
narrator, ten minutes of well-sourced explanation, and a new simple drawing every
few seconds, with one recurring character and the odd handwritten title card.

How a project flows:
  script   each scene is a list of beats; a beat is one sentence of narration plus
           the picture for it (see longform.DOODLE_RULES). ``narration`` is the
           beats joined, so voice, captions and chapters work unchanged.
  voice    as usual; ``fit_duration`` then nudges the pace so the video lands in
           the 9-11 minute window.
  images   one image per beat that is not a title card, drawn on Kaggle's free GPU
           (kaggle_images.py), on this machine's GPU, or as placeholders.
           Raw images land in doodle/raw/s<scene>_b<beat>.png.
  frames   ``compose_frames`` fits each image to the video frame, adds the
           handwritten label, and draws title cards; doodle/frames/.
  plan     ``clip_plan`` times every beat from the narration's word timings, so the
           picture changes exactly when the sentence does.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageStat

from .config import ROOT
from .fonts import _WINDOWS_FONTS, find_font
from .script import normalize_scenes
from .util import log, read_json, write_json

BACKENDS = ("kaggle", "diffusion", "placeholder")
BACKEND_LABELS = {
    "kaggle": "Kaggle: SDXL on a free cloud GPU (about 20-40 min for a whole video)",
    "diffusion": "This PC's GPU: Stable Diffusion 1.5 (about 45 s per image, hours per video)",
    "placeholder": "Placeholders: sketch boxes with the picture text, for testing",
}

DEFAULTS = {
    "images": "kaggle",
    "style": "simple hand-drawn doodle cartoon, thick black outlines, flat colors, minimal shading, "
             "plain white background",
    "negative": "photo, realistic, 3d render, text, letters, words, watermark, signature, logo, "
                "gradient background, blurry, deformed, extra limbs, extra fingers",
    "font_path": "",            # handwriting font; empty = Ink Free / Segoe Print / Comic Sans
    "labels": True,             # draw the beat's handwritten label on its picture
    "card_color": "#8a2b1d",    # title cards: dark red ink on white
    "label_color": "#1f1f1f",
    "fit_minutes": [9, 11],     # nudge the narration pace into this window
    "max_speed_change": 0.12,   # never more than 12% faster or slower
    "min_beat_seconds": 1.4,    # a picture stays up at least this long
    "motion_strength": 0.04,    # how far a drawing drifts over its shot (photos use motion.strength)
    "image_size": [1344, 768],  # placeholder size; Kaggle's size is in kaggle:
}

HANDWRITING = [_WINDOWS_FONTS / "Inkfree.ttf", _WINDOWS_FONTS / "segoepr.ttf", _WINDOWS_FONTS / "comic.ttf",
               Path("/System/Library/Fonts/Supplemental/Chalkboard.ttc"),
               Path("/System/Library/Fonts/Supplemental/Comic Sans MS.ttf"),
               Path("/usr/share/fonts/truetype/comic-neue/ComicNeue-Regular.ttf")]


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update(cfg.get("doodle", {}) or {})
    return merged


def is_doodle(script: dict | None) -> bool:
    """A doodle project, or a Short cut from one: its scenes carry beats."""
    scenes = (script or {}).get("scenes") or []
    return bool(scenes) and all(isinstance(s.get("beats"), list) and s["beats"] for s in scenes)


# ---------------------------------------------------------------------------
# Beats in the script
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_beat(raw) -> dict | None:
    if isinstance(raw, str):
        raw = {"text": raw}
    if not isinstance(raw, dict):
        return None
    text = _clean(raw.get("text"))
    if not text:
        return None
    card = _clean(raw.get("card")).upper()
    return {
        "text": text,
        "picture": _clean(raw.get("picture")) or text,
        "character": bool(raw.get("character")),
        "label": _clean(raw.get("label")).upper()[:40],
        "card": " ".join(card.split()[:6]),
        "take": int(raw.get("take", 0) or 0),
    }


def split_sentences(text: str, longest: int = 18) -> list[str]:
    """Narration cut into beat-sized pieces: sentences, long ones split at a comma."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", _clean(text)) if p]
    pieces: list[str] = []
    for part in parts:
        words = part.split()
        while len(words) > longest:
            # Split at the comma nearest the middle, else in the middle.
            commas = [i + 1 for i, w in enumerate(words[:-3]) if w.endswith((",", ";", ":")) and i >= 3]
            cut = min(commas, key=lambda i: abs(i - len(words) / 2)) if commas else len(words) // 2
            pieces.append(" ".join(words[:cut]))
            words = words[cut:]
        pieces.append(" ".join(words))
    # A two-word fragment is not worth its own picture; fold it into the one before.
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece.split()) < 4:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def beats_from_narration(narration: str, pictures: list[str], fallback_picture: str) -> list[dict]:
    """Beats for narration that has none (or was edited by hand): one per sentence."""
    beats = []
    for k, sentence in enumerate(split_sentences(narration)):
        picture = pictures[k] if k < len(pictures) and pictures[k] else (fallback_picture or sentence)
        beats.append(normalize_beat({"text": sentence, "picture": picture}))
    return [b for b in beats if b]


def normalize_doodle_scenes(raw_scenes) -> list[dict]:
    """Scene dicts with ``beats`` and the usual fields (narration = the beats joined)."""
    if isinstance(raw_scenes, dict):
        raw_scenes = list(raw_scenes.values())
    scenes = []
    for raw in raw_scenes or []:
        if not isinstance(raw, dict):
            continue
        beats = [b for b in (normalize_beat(x) for x in (raw.get("beats") or [])) if b]
        if not beats and _clean(raw.get("narration")):
            beats = beats_from_narration(raw["narration"], [], _clean(raw.get("visual_description")))
        if not beats:
            continue
        drawn = [b for b in beats if not b["card"]] or beats
        base = normalize_scenes([{
            "narration": " ".join(b["text"] for b in beats),
            "visual_description": drawn[0]["picture"],
            "visual_query": " ".join(drawn[0]["picture"].split()[:5]),
            "on_screen_text": next((b["card"] for b in beats if b["card"]), ""),
        }])
        if base:
            base[0]["beats"] = beats
            scenes.append(base[0])
    return scenes


def sync_beats(script: dict) -> bool:
    """Keep beats and narration in step after a hand edit of script.json; True if changed.

    Narration is what gets spoken, so when it no longer matches its beats the beats
    are rebuilt from it, keeping the old pictures in order.
    """
    changed = False
    for scene in script.get("scenes", []):
        beats = [b for b in (normalize_beat(x) for x in (scene.get("beats") or [])) if b]
        narration = _clean(scene.get("narration"))
        if beats and _clean(" ".join(b["text"] for b in beats)) == narration:
            if beats != scene.get("beats"):
                scene["beats"] = beats
                changed = True
            continue
        pictures = [b["picture"] for b in beats if not b["card"]]
        scene["beats"] = beats_from_narration(narration, pictures, scene.get("visual_description", ""))
        changed = True
    return changed


def beat_id(scene: int, beat: int) -> str:
    return f"s{scene:02d}_b{beat:02d}"


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def raw_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "doodle" / "raw"


def frames_dir(out_dir: Path) -> Path:
    return Path(out_dir) / "doodle" / "frames"


def raw_path(out_dir: Path, scene: int, beat: int) -> Path:
    return raw_dir(out_dir) / f"{beat_id(scene, beat)}.png"


def _source(out_dir: Path, script: dict) -> tuple[Path, int]:
    """Where a project's raw images live: its own folder, or the long video it was cut from."""
    src = script.get("doodle_source") or {}
    if src.get("dir"):
        return Path(src["dir"]), int(src.get("first_scene", 0))
    return Path(out_dir), 0


def _words(text: str, limit: int) -> str:
    return " ".join(str(text or "").split()[:limit])


MAIN_CHARACTER = re.compile(r"\b(?:the |our )?main character\b", re.IGNORECASE)


def _subject(beat: dict, script: dict) -> str:
    character = _words(script.get("main_character"), 14)
    picture = _words(beat.get("picture"), 30)
    # Scripts often say "Main character sipping coffee"; the image model needs the look itself.
    if character and MAIN_CHARACTER.search(picture):
        return MAIN_CHARACTER.sub(_words(script.get("main_character"), 20), picture, count=1)
    parts = [picture]
    if beat.get("character") and character:
        parts.append(character)
    return ", ".join(p for p in parts if p)


def build_prompt(beat: dict, script: dict, cfg: dict) -> str:
    """Picture and character first (CLIP drops what comes after ~77 tokens), then the style."""
    style = _words(script.get("art_style") or settings(cfg)["style"], 18)
    return f"{_subject(beat, script)}, {style}"


def character_prompt(script: dict, cfg: dict) -> str:
    """The reference drawing of the recurring character that keeps it consistent on Kaggle."""
    character = _words(script.get("main_character"), 20)
    if not character:
        return ""
    style = _words(script.get("art_style") or settings(cfg)["style"], 18)
    return f"{character}, full body, standing, front view, friendly expression, centered, {style}"


def seed_for(script: dict, scene: int, beat: int, take: int = 0) -> int:
    base = int(hashlib.sha256(str(script.get("title", "")).encode("utf-8")).hexdigest()[:8], 16)
    return (base + scene * 1009 + beat * 101 + take * 7919) % (2 ** 31)


def image_jobs(out_dir: Path, script: dict, cfg: dict, only_missing: bool = True) -> list[dict]:
    """One job per beat that needs a drawing (title cards are drawn locally, not generated)."""
    jobs = []
    for i, scene in enumerate(script.get("scenes", [])):
        for k, beat in enumerate(scene.get("beats") or []):
            if beat.get("card"):
                continue
            if only_missing and raw_path(out_dir, i, k).exists():
                continue
            jobs.append({"id": beat_id(i, k), "prompt": build_prompt(beat, script, cfg),
                         "subject": _subject(beat, script),
                         "seed": seed_for(script, i, k, int(beat.get("take", 0))),
                         "character": bool(beat.get("character"))})
    return jobs


def image_status(out_dir: Path, script: dict) -> dict:
    """How many drawings the project needs and has."""
    src, offset = _source(out_dir, script)
    needed = done = cards = 0
    missing: list[str] = []
    for i, scene in enumerate(script.get("scenes", [])):
        for k, beat in enumerate(scene.get("beats") or []):
            if beat.get("card"):
                cards += 1
                continue
            needed += 1
            if raw_path(src, i + offset, k).exists():
                done += 1
            else:
                missing.append(beat_id(i, k))
    return {"needed": needed, "done": done, "cards": cards, "missing": missing,
            "beats": needed + cards}


def mark_for_redo(out_dir: Path, script: dict, ids: list[str]) -> int:
    """Delete these beats' drawings and bump their seed, so the next job draws new takes."""
    count = 0
    wanted = set(ids)
    for i, scene in enumerate(script.get("scenes", [])):
        for k, beat in enumerate(scene.get("beats") or []):
            if beat_id(i, k) in wanted and not beat.get("card"):
                beat["take"] = int(beat.get("take", 0)) + 1
                raw_path(out_dir, i, k).unlink(missing_ok=True)
                count += 1
    return count


def generate_images(out_dir: Path, script: dict, cfg: dict, backend: str) -> dict:
    """Draw every missing image with the chosen backend; returns image_status afterwards."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown image backend '{backend}'; choose one of {', '.join(BACKENDS)}")
    if script.get("doodle_source"):
        return image_status(out_dir, script)   # a cut reuses its parent's drawings
    jobs = image_jobs(out_dir, script, cfg)
    raw_dir(out_dir).mkdir(parents=True, exist_ok=True)
    if not jobs:
        log("doodle: every drawing already exists")
        return image_status(out_dir, script)
    log(f"doodle: {len(jobs)} drawings to make via {backend}")
    if backend == "kaggle":
        from .kaggle_images import run_job
        run_job(out_dir, jobs, character_prompt(script, cfg), cfg)
    elif backend == "diffusion":
        _draw_local(out_dir, jobs, script, cfg)
    else:
        size = tuple(settings(cfg)["image_size"])
        for job in jobs:
            draw_placeholder(job["prompt"], raw_dir(out_dir) / f"{job['id']}.png", size, job["character"])
    status = image_status(out_dir, script)
    log(f"doodle: {status['done']}/{status['needed']} drawings ready")
    return status


def _draw_local(out_dir: Path, jobs: list[dict], script: dict, cfg: dict) -> None:
    from . import diffusion
    if not diffusion.available():
        raise RuntimeError(diffusion.install_hint())
    context = {"title": script.get("title", ""), "art_style": script.get("art_style") or settings(cfg)["style"],
               "content_type": "fact"}
    for n, job in enumerate(jobs, 1):
        scene, beat = (int(x[1:]) for x in job["id"].split("_"))
        log(f"doodle: local drawing {n}/{len(jobs)} ({job['id']})")
        # The panel generator adds the art style itself, so it gets the subject alone.
        panel = diffusion.generate_panel({"visual_description": job["subject"]}, out_dir, 1000 + scene * 100 + beat,
                                         cfg, context, variant=job["seed"] % 1000)
        shutil.copyfile(panel, raw_dir(out_dir) / f"{job['id']}.png")


# ---------------------------------------------------------------------------
# Drawing on the page: fonts, placeholders, cards, labels
# ---------------------------------------------------------------------------

def handwriting_font(cfg: dict) -> Path | None:
    preferred = str(settings(cfg).get("font_path") or "")
    candidates = ([Path(preferred)] if preferred else []) + sorted((ROOT / "assets" / "fonts").glob("*.ttf")) + HANDWRITING
    found = next((p for p in candidates if p.exists()), None)
    return found or find_font(bold=True)


def _font(path: Path | None, size: int) -> ImageFont.ImageFont:
    if path:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def draw_placeholder(text: str, path: Path, size: tuple[int, int], character: bool = False,
                     cfg: dict | None = None) -> Path:
    """A sketch-box stand-in for a drawing: the picture text, and a stick figure for the character."""
    w, h = size
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((24, 24, w - 24, h - 24), radius=28, outline=(40, 40, 40), width=5)
    if character:
        cx, cy, r = w // 6, h // 2 - 60, 44
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline="black", width=6)
        d.line((cx, cy + r, cx, cy + r + 150), fill="black", width=6)
        d.line((cx - 60, cy + r + 50, cx + 60, cy + r + 50), fill="black", width=6)
        d.line((cx, cy + r + 150, cx - 50, cy + r + 240), fill="black", width=6)
        d.line((cx, cy + r + 150, cx + 50, cy + r + 240), fill="black", width=6)
    font = _font(handwriting_font(cfg or {}), max(22, h // 22))
    left = w // 3 if character else 70
    lines = textwrap.wrap(text, width=max(18, int((w - left - 70) / (h // 22 * 0.55))))[:9]
    y = (h - len(lines) * int(h / 18)) // 2
    for line in lines:
        d.text((left, y), line, font=font, fill=(90, 90, 90))
        y += int(h / 18)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=True)
    return path


def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Fill the frame when the shapes are close; otherwise fit inside a white page (16:9 art in a Short)."""
    img = img.convert("RGB")
    ratio = (img.width / img.height) / (w / h)
    if 0.8 <= ratio <= 1.25:
        scale = max(w / img.width, h / img.height)
        big = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
        left, top = (big.width - w) // 2, (big.height - h) // 2
        return big.crop((left, top, left + w, top + h))
    scale = min(w / img.width, h / img.height)
    small = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    page = Image.new("RGB", (w, h), "white")
    page.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
    return page


def _wavy_underline(d: ImageDraw.ImageDraw, x0: int, x1: int, y: int, color: str, width: int) -> None:
    pts, step = [], max(12, (x1 - x0) // 24)
    for n, x in enumerate(range(x0, x1 + 1, step)):
        pts.append((x, y + (4 if n % 2 else -4)))
    if len(pts) > 1:
        d.line(pts, fill=color, width=width, joint="curve")


def draw_card(text: str, w: int, h: int, cfg: dict) -> Image.Image:
    """A full-screen handwritten title card: dark red ink, a wavy underline, nothing else."""
    s = settings(cfg)
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    font_path = handwriting_font(cfg)
    size = int(min(w, h) * 0.14)
    words = text.split()
    while True:
        font = _font(font_path, size)
        per_line = max(1, min(len(words), 3 if w >= h else 2))
        lines = [" ".join(words[i:i + per_line]) for i in range(0, len(words), per_line)]
        widths = [d.textbbox((0, 0), ln, font=font)[2] for ln in lines]
        if max(widths, default=0) <= w * 0.82 or size <= 40:
            break
        size -= 6
    line_h = int(size * 1.25)
    y = (h - line_h * len(lines)) // 2
    for ln, lw in zip(lines, widths):
        d.text(((w - lw) // 2, y), ln, font=font, fill=s["card_color"])
        y += line_h
    _wavy_underline(d, int(w * 0.32), int(w * 0.68), y + size // 6, s["card_color"], max(4, size // 14))
    return img


def _emptiest_corner(img: Image.Image, box_w: int, box_h: int, margin: int) -> tuple[int, int]:
    """Top-left of the most blank corner region (brightest and flattest), for a label."""
    w, h = img.size
    gray = img.convert("L")
    spots = [(margin, margin), (w - box_w - margin, margin),
             (margin, h - box_h - margin), (w - box_w - margin, h - box_h - margin)]
    best, best_score = spots[0], -1e9
    for x, y in spots:
        region = gray.crop((max(0, x), max(0, y), min(w, x + box_w), min(h, y + box_h)))
        stat = ImageStat.Stat(region)
        score = stat.mean[0] - 2.0 * stat.stddev[0]
        if score > best_score:
            best, best_score = (x, y), score
    return best


def draw_label(img: Image.Image, text: str, cfg: dict) -> Image.Image:
    """A short handwritten note in the emptiest corner of the drawing, underlined."""
    s = settings(cfg)
    w, h = img.size
    d = ImageDraw.Draw(img)
    size = int(h / 13)
    font = _font(handwriting_font(cfg), size)
    box = d.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    margin = int(h * 0.06)
    x, y = _emptiest_corner(img, tw + 20, th + 30, margin)
    d.text((x + 10, y), text, font=font, fill=s["label_color"])
    _wavy_underline(d, x + 10, x + 10 + tw, y + th + int(size * 0.35), s["label_color"], max(3, size // 16))
    return img


def compose_frames(out_dir: Path, script: dict, cfg: dict) -> list[list[Path]]:
    """Per scene, per beat: the finished frame at the video size (cards drawn, labels added).

    Frames are cached by their inputs, so re-rendering after one redraw only redoes that beat.
    """
    s = settings(cfg)
    w, h = cfg["video"]["resolution"]
    src, offset = _source(out_dir, script)
    fdir = frames_dir(out_dir)
    fdir.mkdir(parents=True, exist_ok=True)
    font = handwriting_font(cfg)
    missing = []
    frames: list[list[Path]] = []
    for i, scene in enumerate(script.get("scenes", [])):
        row = []
        for k, beat in enumerate(scene.get("beats") or []):
            raw = raw_path(src, i + offset, k)
            if not beat.get("card") and not raw.exists():
                missing.append(beat_id(i, k))
                row.append(raw)
                continue
            label = beat.get("label", "") if s["labels"] else ""
            stamp = f"{raw.stat().st_mtime_ns}:{raw.stat().st_size}" if not beat.get("card") else ""
            key = "|".join([beat.get("card", ""), label, stamp, f"{w}x{h}", str(font), s["card_color"],
                            s["label_color"]])
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
            path = fdir / f"{beat_id(i, k)}_{digest}.png"
            if not path.exists():
                for stale in fdir.glob(f"{beat_id(i, k)}_*.png"):
                    stale.unlink(missing_ok=True)
                if beat.get("card"):
                    img = draw_card(beat["card"], w, h, cfg)
                else:
                    with Image.open(raw) as im:
                        img = _fit(im, w, h)
                    if label:
                        img = draw_label(img, label, cfg)
                img.save(path, format="PNG", optimize=True)
            row.append(path)
        frames.append(row)
    if missing:
        raise RuntimeError(f"{len(missing)} drawings are missing (first: {', '.join(missing[:5])}); "
                           "draw them before rendering")
    return frames


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def beat_durations(beats: list[dict], audio, min_seconds: float = 1.4) -> list[float]:
    """Seconds per beat so each picture changes when its sentence starts."""
    total = float(audio.duration)
    counts = [max(1, len(b["text"].split())) for b in beats]
    if len(beats) == 1:
        return [total]
    words = audio.words or []
    n_words = sum(counts)
    starts = [0.0]
    cum = 0
    for c in counts[:-1]:
        cum += c
        if words:
            idx = min(len(words) - 1, round(cum * len(words) / n_words))
            starts.append(float(words[idx][1]))
        else:
            starts.append(total * cum / n_words)
    starts.append(total)
    # Starts must rise; a bad timing must never produce a negative shot.
    for j in range(1, len(starts)):
        starts[j] = min(max(starts[j], starts[j - 1]), total)
    durs = [starts[j + 1] - starts[j] for j in range(len(beats))]
    # Lend time to beats that would flash by, from the longest neighbour.
    for j, d in enumerate(durs):
        if d >= min_seconds:
            continue
        neighbours = [x for x in (j - 1, j + 1) if 0 <= x < len(durs)]
        donor = max(neighbours, key=lambda x: durs[x])
        give = min(min_seconds - d, max(0.0, durs[donor] - min_seconds))
        durs[j] += give
        durs[donor] -= give
    drift = total - sum(durs)
    durs[-1] += drift
    return durs


GENTLE_MOVES = ("zoom_in", "pan_right", "zoom_out", "pan_left")


def clip_plan(script: dict, audios: list, frames: list[list[Path]], cfg: dict) -> list[list[tuple]]:
    """(frame, seconds, shot) per beat; a slow drift on drawings, a still title card."""
    s = settings(cfg)
    plan = []
    n = 0
    for scene, audio, row in zip(script.get("scenes", []), audios, frames):
        beats = scene.get("beats") or []
        durs = beat_durations(beats, audio, float(s["min_beat_seconds"]))
        shots = []
        for beat, seconds, frame in zip(beats, durs, row):
            if beat.get("card"):
                shot = {"move": "zoom_in", "zoom": 1.0, "focus": [0.5, 0.5], "strength": 0.02}
            else:
                shot = {"move": GENTLE_MOVES[n % len(GENTLE_MOVES)], "zoom": 1.02, "focus": [0.5, 0.5],
                        "strength": float(s["motion_strength"])}
            n += 1
            shots.append((frame, seconds, shot))
        plan.append(shots)
    return plan


def fit_duration(out_dir: Path, audios: list, cfg: dict) -> list:
    """Re-time the narration evenly so the video lands in ``fit_minutes``; returns the audios."""
    from .voice import apply_scene_speed
    s = settings(cfg)
    low, high = (float(x) * 60 for x in s["fit_minutes"])
    saved = [_audio_meta(out_dir, i) for i in range(len(audios))]
    originals = [float(m.get("orig_duration", a.duration)) for m, a in zip(saved, audios)]
    total = sum(originals)
    if not total:
        return audios
    target = min(max(total, low), high)
    limit = float(s["max_speed_change"])
    speed = min(max(total / target, 1 - limit), 1 + limit)
    if abs(speed - 1.0) < 0.005:
        speed = 1.0
    current = [float(m.get("speed", 1.0)) for m in saved]
    if all(abs(c - speed) < 0.005 for c in current):
        return audios
    if speed == 1.0:
        log(f"doodle: narration is {total / 60:.1f} min, inside the {low / 60:.0f}-{high / 60:.0f} min window")
    else:
        log(f"doodle: narration is {total / 60:.1f} min; re-timing to {speed:.2f}x "
            f"for about {total / speed / 60:.1f} min")
    fitted = [apply_scene_speed(Path(out_dir), i, speed) for i in range(len(audios))]
    final = sum(a.duration for a in fitted)
    if not low - 1 <= final <= high + 1:
        log(f"doodle: still {final / 60:.1f} min after the largest allowed change; "
            "lengthen or shorten the script for a 9-11 minute video")
    return fitted


def _audio_meta(out_dir: Path, index: int) -> dict:
    meta = Path(out_dir) / "audio" / f"scene_{index:02d}.json"
    try:
        return read_json(meta) if meta.exists() else {}
    except (OSError, ValueError):
        return {}


def write_beat_index(out_dir: Path, script: dict) -> None:
    """doodle/beats.json: every beat with its id, for reading the project by hand."""
    rows = []
    for i, scene in enumerate(script.get("scenes", [])):
        for k, beat in enumerate(scene.get("beats") or []):
            rows.append({"id": beat_id(i, k), **beat})
    (Path(out_dir) / "doodle").mkdir(parents=True, exist_ok=True)
    write_json(Path(out_dir) / "doodle" / "beats.json", rows)

