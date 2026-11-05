"""Camera moves for still images, so a photo or AI panel never just sits there.

A still that is held for a whole scene reads as a slideshow. Instead each scene
is cut into short shots: a slow push in, a pull back, a pan across, each from a
different part of the picture, and when the scene has more than one usable
still they alternate. Videos keep their own motion and are untouched here.

A shot is a small dict carried in the clip plan next to the path and duration:
    {"move": "pan_left", "zoom": 1.22, "focus": [0.5, 0.3]}
``zoom`` is how far in the camera sits (1.0 = full frame), ``focus`` is the
point of the picture (0..1) the camera is centred on or moves towards.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MOVES = ("zoom_in", "pan_left", "zoom_out", "pan_up", "pan_right", "pan_down")
FOCI = ((0.5, 0.5), (0.5, 0.25), (0.5, 0.75), (0.3, 0.4), (0.7, 0.6))

DEFAULTS = {
    "seconds_per_still": 3.0,   # a new shot about this often while a still is on screen
    "min_shot_seconds": 1.6,    # never cut faster than this
    "strength": 0.18,           # how far a move travels over its shot (fraction of the frame)
    "base_zoom": 1.12,          # how far in pans and pull-backs sit (1.0 = whole picture)
}


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update(cfg.get("motion", {}) or {})
    if "seconds_per_still" in cfg.get("video", {}):
        merged["seconds_per_still"] = cfg["video"]["seconds_per_still"]
    return merged


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def plan_shots(images: list[Path], duration: float, cfg: dict, seed_text: str = "") -> list[tuple]:
    """Split ``duration`` into shots over ``images``; returns (path, seconds, shot) triples.

    The first image is the chosen one and opens the scene. Extra images take
    turns; a lone image is re-framed each shot so it still moves.
    """
    s = settings(cfg)
    images = [Path(p) for p in images if p]
    if not images:
        return []
    per = max(float(s["seconds_per_still"]), 0.5)
    cuts = max(1, round(duration / per))
    while cuts > 1 and duration / cuts < float(s["min_shot_seconds"]):
        cuts -= 1
    seg = duration / cuts
    offset = _seed(seed_text or str(images[0]))
    shots = []
    for k in range(cuts):
        image = images[k % len(images)]
        move = MOVES[(offset + k) % len(MOVES)]
        # Re-framing matters most when the same picture comes round again.
        focus = FOCI[(offset // 7 + k // max(len(images), 1)) % len(FOCI)] if cuts > len(images) else FOCI[(offset + k) % len(FOCI)]
        if move == "zoom_in":
            focus = FOCI[(offset + k) % len(FOCI)]
        shot = {"move": move, "zoom": round(float(s["base_zoom"]), 3), "focus": [focus[0], focus[1]]}
        shots.append((image, seg, shot))
    # Spread rounding so the shots add up to the narration exactly.
    total = sum(x[1] for x in shots)
    if shots and abs(total - duration) > 1e-6:
        path, seconds, shot = shots[-1]
        shots[-1] = (path, seconds + (duration - total), shot)
    return shots


def default_shot() -> dict:
    return {"move": "zoom_in", "zoom": 1.0, "focus": [0.5, 0.5]}


def zoompan_filter(shot: dict | None, width: int, height: int, fps: int, seconds: float,
                   strength: float = DEFAULTS["strength"]) -> str:
    """ffmpeg -vf chain that animates one still for ``seconds`` at the video size.

    zoompan's ``on`` is the output frame index, so every move is an explicit
    function of time instead of an accumulating step: it is deterministic and
    ends exactly where it should.
    """
    shot = dict(default_shot(), **(shot or {}))
    frames = max(int(round(seconds * fps)), 1)
    fx, fy = (float(v) for v in shot.get("focus", (0.5, 0.5)))
    base = max(float(shot.get("zoom", 1.0)), 1.0)
    travel = max(float(strength), 0.02)
    t = f"(on/{frames})"
    move = shot.get("move", "zoom_in")
    if move == "zoom_in":
        z = f"({base}+{travel}*{t})"
        x, y = f"(iw-iw/zoom)*{fx}", f"(ih-ih/zoom)*{fy}"
    elif move == "zoom_out":
        z = f"({base + travel}-{travel}*{t})"
        x, y = f"(iw-iw/zoom)*{fx}", f"(ih-ih/zoom)*{fy}"
    else:
        # Pans sit at a fixed zoom and slide the window; the picture must be larger
        # than the window for that, so pans never drop below a modest zoom.
        zoom = max(base, 1.0 + travel)
        z = f"{zoom}"
        if move == "pan_left":
            x, y = f"(iw-iw/zoom)*(1-{t})", f"(ih-ih/zoom)*{fy}"
        elif move == "pan_right":
            x, y = f"(iw-iw/zoom)*{t}", f"(ih-ih/zoom)*{fy}"
        elif move == "pan_up":
            x, y = f"(iw-iw/zoom)*{fx}", f"(ih-ih/zoom)*(1-{t})"
        else:  # pan_down
            x, y = f"(iw-iw/zoom)*{fx}", f"(ih-ih/zoom)*{t}"
    # Render the move oversized, then scale down: zoompan's integer window
    # positions would otherwise make slow pans judder. 1.5x is the sweet spot
    # between smoothness and render time on a CPU encoder.
    big_w, big_h = (int(round(width * 1.5 / 2)) * 2, int(round(height * 1.5 / 2)) * 2)
    return (f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase,"
            f"crop={big_w}:{big_h},"
            f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={big_w}x{big_h}:fps={fps},"
            f"scale={width}:{height}:flags=lanczos,setsar=1")


def plan_to_json(plan: list) -> list:
    """clip_plan.json form: [[path, seconds, shot?], ...] per scene."""
    out = []
    for scene in plan:
        row = []
        for item in scene:
            path, seconds = item[0], item[1]
            entry = [str(path), round(float(seconds), 2)]
            if len(item) > 2 and item[2]:
                entry.append(item[2])
            row.append(entry)
        out.append(row)
    return out
