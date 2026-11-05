"""Narrator pool: which voices exist, which fit a video, and which one is next.

Thirty videos a month read by the same voice look templated. This module keeps
a catalogue of the free voices (Kokoro's English set on disk, Gemini's prebuilt
set on the free tier), splits them into pools per content type, and rotates
through them so consecutive videos get different narrators. The chosen voice is
written into script.json, so a project keeps its narrator across re-runs, and
into history.json, so the rotation can see what was used recently.
"""
from __future__ import annotations

import hashlib

from . import history

ENGINES = ("kokoro", "gemini", "elevenlabs", "local")

# Kokoro v1.0 English voices with the model card's quality grade. Grades below C
# sound robotic or clipped and are left out of the default pools.
KOKORO_VOICES = {
    "af_heart":    ("American female, warm and clear", "A"),
    "af_bella":    ("American female, expressive", "A-"),
    "af_nicole":   ("American female, soft and intimate", "B-"),
    "af_aoede":    ("American female, bright", "C+"),
    "af_kore":     ("American female, steady", "C+"),
    "af_sarah":    ("American female, friendly", "C+"),
    "af_alloy":    ("American female, neutral", "C"),
    "af_nova":     ("American female, light", "C"),
    "af_sky":      ("American female, airy", "C-"),
    "af_jessica":  ("American female", "D"),
    "af_river":    ("American female", "D"),
    "am_fenrir":   ("American male, deep", "C+"),
    "am_michael":  ("American male, calm", "C+"),
    "am_puck":     ("American male, upbeat", "C+"),
    "am_adam":     ("American male", "F+"),
    "am_echo":     ("American male", "D"),
    "am_eric":     ("American male", "D"),
    "am_liam":     ("American male", "D"),
    "am_onyx":     ("American male", "D"),
    "am_santa":    ("American male, jolly", "D-"),
    "bf_emma":     ("British female, warm", "B-"),
    "bf_isabella": ("British female, poised", "C"),
    "bf_alice":    ("British female", "D"),
    "bf_lily":     ("British female", "D"),
    "bm_george":   ("British male, measured", "C"),
    "bm_fable":    ("British male, storyteller", "C"),
    "bm_lewis":    ("British male", "D+"),
    "bm_daniel":   ("British male", "D"),
}

# Gemini TTS prebuilt voices (free tier) with Google's one-word descriptor.
GEMINI_VOICES = {
    "Zephyr": "Bright", "Puck": "Upbeat", "Charon": "Informative", "Kore": "Firm",
    "Fenrir": "Excitable", "Leda": "Youthful", "Orus": "Firm", "Aoede": "Breezy",
    "Callirrhoe": "Easy-going", "Autonoe": "Bright", "Enceladus": "Breathy", "Iapetus": "Clear",
    "Umbriel": "Easy-going", "Algieba": "Smooth", "Despina": "Smooth", "Erinome": "Clear",
    "Algenib": "Gravelly", "Rasalgethi": "Informative", "Laomedeia": "Upbeat", "Achernar": "Soft",
    "Alnilam": "Firm", "Schedar": "Even", "Gacrux": "Mature", "Pulcherrima": "Forward",
    "Achird": "Friendly", "Zubenelgenubi": "Casual", "Vindemiatrix": "Gentle", "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable", "Sulafat": "Warm",
}

DEFAULT_POOLS = {
    "kokoro": {
        "story": ["af_heart", "af_bella", "bf_emma", "bm_fable", "bm_george", "am_michael",
                  "af_nicole", "am_fenrir"],
        "fact": ["af_nicole", "am_puck", "bf_isabella", "af_kore", "af_sarah", "am_michael",
                 "bm_george", "af_aoede"],
    },
    "gemini": {
        "story": ["Sulafat", "Aoede", "Vindemiatrix", "Enceladus", "Gacrux", "Achernar",
                  "Algieba", "Despina"],
        "fact": ["Charon", "Rasalgethi", "Sadaltager", "Kore", "Iapetus", "Schedar",
                 "Zephyr", "Orus"],
    },
}

DEFAULT_VOICE = {"kokoro": "af_heart", "gemini": "Sulafat", "elevenlabs": "", "local": ""}


def describe(engine: str, name: str) -> str:
    """Short human label for a voice, e.g. 'bm_fable (British male, storyteller)'."""
    if not name:
        return {"local": "system default voice", "elevenlabs": "ELEVENLABS_VOICE_ID from .env"}.get(engine, "default")
    if engine == "kokoro" and name in KOKORO_VOICES:
        return f"{name} ({KOKORO_VOICES[name][0]})"
    if engine == "gemini" and name in GEMINI_VOICES:
        return f"{name} ({GEMINI_VOICES[name]})"
    return name


def pools(cfg: dict) -> dict:
    """Voice pools per engine and content type, config overriding the defaults."""
    merged = {k: {t: list(v) for t, v in sub.items()} for k, sub in DEFAULT_POOLS.items()}
    for engine, sub in (cfg.get("voice", {}).get("pools") or {}).items():
        if not isinstance(sub, dict):
            continue
        merged.setdefault(engine, {})
        for content_type, names in sub.items():
            if isinstance(names, list) and names:
                merged[engine][content_type] = [str(n) for n in names]
    return merged


def pool_for(cfg: dict, engine: str, content_type: str) -> list[str]:
    sub = pools(cfg).get(engine, {})
    names = sub.get(content_type) or [n for group in sub.values() for n in group]
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def all_voices(cfg: dict, engine: str) -> list[str]:
    """Every selectable voice for an engine, pool members first."""
    names = [n for group in pools(cfg).get(engine, {}).values() for n in group]
    catalogue = KOKORO_VOICES if engine == "kokoro" else GEMINI_VOICES if engine == "gemini" else {}
    names += [n for n in catalogue if n not in names]
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def fixed_voice(cfg: dict, engine: str) -> str:
    v = cfg.get("voice", {})
    return str(v.get(f"{engine}_voice", "") or DEFAULT_VOICE.get(engine, ""))


def usage(engine: str) -> dict[str, str]:
    """Most recent creation date per voice name from history.json."""
    last: dict[str, str] = {}
    for entry in history.load():
        voice = entry.get("voice") or {}
        if voice.get("engine") == engine and voice.get("name"):
            last[voice["name"]] = max(last.get(voice["name"], ""), entry.get("created", ""))
    return last


def _recency(engine: str) -> dict[str, int]:
    """Position in history of each voice's latest use; later means more recent."""
    seen: dict[str, int] = {}
    for index, entry in enumerate(history.load()):
        voice = entry.get("voice") or {}
        if voice.get("engine") == engine and voice.get("name"):
            seen[voice["name"]] = index
    return seen


def pick_voice(cfg: dict, engine: str, content_type: str, seed_text: str = "") -> str:
    """Next narrator for a new video: least recently used in the pool, ties by title hash."""
    if engine not in ("kokoro", "gemini"):
        return fixed_voice(cfg, engine)
    if not cfg.get("voice", {}).get("rotate", True):
        return fixed_voice(cfg, engine)
    pool = pool_for(cfg, engine, content_type)
    if not pool:
        return fixed_voice(cfg, engine)
    last = _recency(engine)
    tie = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    ranked = sorted(pool, key=lambda n: (last.get(n, -1), (pool.index(n) + tie) % len(pool)))
    return ranked[0]


def resolve_voice(script: dict, cfg: dict, engine: str, requested: str | None = None) -> dict:
    """The voice a project should use, honouring an explicit request, then the script's own.

    Returns {"engine": ..., "name": ...}. The caller persists it into script.json.
    """
    current = script.get("voice") if isinstance(script.get("voice"), dict) else {}
    if requested:
        name = requested
    elif current.get("engine") == engine and current.get("name") is not None:
        name = str(current.get("name", ""))
    else:
        name = pick_voice(cfg, engine, script.get("content_type", "story"), script.get("title", ""))
    if engine == "kokoro" and name and name not in KOKORO_VOICES:
        raise ValueError(f"unknown Kokoro voice '{name}'; run: python run.py voices")
    if engine == "gemini" and name and name not in GEMINI_VOICES:
        raise ValueError(f"unknown Gemini voice '{name}'; run: python run.py voices")
    return {"engine": engine, "name": name}
