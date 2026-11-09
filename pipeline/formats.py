"""Video formats: what kind of video a project is, and the settings that follow from it.

  short  vertical 9:16, under a minute, a handful of scenes (YouTube Shorts)
  long   16:9, around ten minutes, chaptered narration with timestamps
  doodle 16:9, 9-11 minutes, a hand-drawn cartoon for every sentence (images drawn on Kaggle)
  lofi   16:9, a looping animated scene over generated music, no narration

``config.yaml`` has one base configuration (tuned for Shorts) and a ``formats:``
section with per-format overrides. ``apply(cfg, name)`` returns a copy of the
config with those overrides merged in, so every stage keeps reading the same
keys (``video.resolution``, ``captions.size`` ...) and never needs to know
which format it is rendering. The chosen format is written into ``script.json``
so a project renders the same way when it is resumed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_FORMAT = "short"

# Shipped defaults; config.yaml's ``formats:`` section overrides any key.
BUILTIN = {
    "short": {
        "label": "Short: vertical, under a minute",
        "video": {"resolution": [1080, 1920], "target_minutes": 0.8, "max_scenes": 6},
        "captions": {"size": 84, "words_per_line": 4},
    },
    "long": {
        "label": "Long: 16:9, about ten minutes, with chapters",
        "video": {"resolution": [1920, 1080], "target_minutes": 10, "max_scenes": 0,
                  "chapters": 6, "scenes_per_chapter": 6},
        "captions": {"size": 54, "words_per_line": 5},
        "visuals": {"rank_batch_scenes": 6, "rank_max_requests": 6, "candidates_per_scene": 5},
        "diffusion": {"width": 896, "height": 512},
    },
    "doodle": {
        "label": "Doodle explainer: 16:9, 9 to 11 minutes, a new drawing every sentence",
        "video": {"resolution": [1920, 1080], "target_minutes": 10, "max_scenes": 0,
                  "chapters": 6, "scenes_per_chapter": 5},
        "captions": {"enabled": False, "size": 54, "words_per_line": 5},
        "motion": {"strength": 0.04, "base_zoom": 1.02},
        "diffusion": {"width": 896, "height": 512},
    },
    "lofi": {
        "label": "Lofi: 16:9 looping scene over generated music",
        "video": {"resolution": [1920, 1080], "fps": 24, "max_scenes": 1},
        "captions": {"enabled": False},
        "diffusion": {"width": 896, "height": 512},
    },
}
# Sections a format may override; anything else in a profile is informational.
SECTIONS = ("video", "captions", "visuals", "diffusion", "music", "lofi", "youtube", "motion",
            "doodle", "kaggle")


def names(cfg: dict | None = None) -> list[str]:
    extra = [k for k in ((cfg or {}).get("formats") or {}) if k not in BUILTIN]
    return list(BUILTIN) + extra


def label(cfg: dict | None, name: str) -> str:
    return str(profile(cfg or {}, name).get("label") or name)


def profile(cfg: dict, name: str) -> dict:
    """The format's overrides: shipped defaults under the user's config.yaml entries."""
    merged = copy.deepcopy(BUILTIN.get(name, {}))
    user = ((cfg.get("formats") or {}).get(name)) or {}
    for key, value in user.items():
        if key in SECTIONS and isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
        else:
            merged[key] = value
    return merged


def is_landscape(cfg: dict) -> bool:
    w, h = cfg["video"]["resolution"]
    return w >= h


def apply(cfg: dict, name: str | None) -> dict:
    """A copy of ``cfg`` with the format's overrides merged in and ``cfg['format']`` set."""
    name = name or DEFAULT_FORMAT
    if name not in names(cfg):
        raise ValueError(f"unknown video format '{name}'; choose one of {', '.join(names(cfg))}")
    out = copy.deepcopy(cfg)
    for section, values in profile(cfg, name).items():
        if section in SECTIONS and isinstance(values, dict):
            out.setdefault(section, {}).update(copy.deepcopy(values))
    out["format"] = name
    if is_landscape(out):
        # The shipped art styles ask for a vertical frame; widen them for 16:9 formats.
        diff = out.get("diffusion") or {}
        if isinstance(diff.get("style"), str):
            diff["style"] = _widen(diff["style"])
        if isinstance(diff.get("styles"), list):
            diff["styles"] = [_widen(str(s)) for s in diff["styles"]]
    return out


def _widen(style: str) -> str:
    return style.replace("vertical composition", "wide cinematic composition")


def of_script(script: dict | None) -> str:
    """The format a project was made in (older projects are Shorts)."""
    name = str((script or {}).get("format") or "").strip().lower()
    return name or DEFAULT_FORMAT


def for_project(cfg: dict, out_dir: Path) -> dict:
    """``apply`` with the format recorded in ``out_dir/script.json``."""
    path = Path(out_dir) / "script.json"
    name = DEFAULT_FORMAT
    if path.exists():
        try:
            name = of_script(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return apply(cfg, name if name in names(cfg) else DEFAULT_FORMAT)
