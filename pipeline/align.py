"""Word timing for voices that do not report it: a local Whisper pass over the narration.

Kokoro tells us when each phoneme is spoken, but Gemini TTS, ElevenLabs without
alignment data, and the system voice only hand back audio. For those, this module
transcribes the clip with faster-whisper (word timestamps on, the script as the
decoding prompt) and maps the recognised words back onto the script's own words,
so captions keep the script's spelling and punctuation while following the real
speech. Runs on the CPU in a few seconds per scene; nothing leaves the machine.
"""
from __future__ import annotations

import difflib
import importlib.util
import os
import re
import subprocess
import threading
from pathlib import Path

from .config import ROOT
from .util import log

DEFAULTS = {
    "align": True,                 # use Whisper for engines without model timings
    "align_model": "small.en",     # tiny.en / base.en / small.en / medium.en; downloaded once
    "align_device": "cpu",         # cpu is fine: a 10 s scene takes a few seconds
    "align_compute": "int8",
    "align_cache_dir": "cache/asr",
}
MIN_MATCH_RATIO = 0.5   # below this Whisper heard something else; fall back to estimates

_MODEL = None
_MODEL_KEY = None
_LOCK = threading.RLock()  # one Whisper model, one transcription at a time


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in (cfg.get("captions", {}) or {}).items() if k in DEFAULTS})
    return merged


def available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def install_hint() -> str:
    return ("Accurate captions for this voice need faster-whisper. Run:\n"
            "  .venv\\Scripts\\pip install faster-whisper")


def _load(cfg: dict):
    with _LOCK:
        return _load_unlocked(cfg)


def _load_unlocked(cfg: dict):
    global _MODEL, _MODEL_KEY
    s = settings(cfg)
    key = (s["align_model"], s["align_device"], s["align_compute"])
    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from faster_whisper import WhisperModel
    cache_dir = ROOT / s["align_cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"captions: loading Whisper {s['align_model']} on {s['align_device']} (first run downloads it)")
    _MODEL = WhisperModel(s["align_model"], device=s["align_device"], compute_type=s["align_compute"],
                          download_root=str(cache_dir))
    _MODEL_KEY = key
    return _MODEL


def _pcm(audio: Path):
    """Decode with ffmpeg to 16 kHz mono float32, the form Whisper wants.

    Decoding here keeps faster-whisper's PyAV dependency out of the picture; the
    two have disagreed about keyword arguments across releases.
    """
    import numpy as np
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(audio),
         "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
        capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower().replace("’", "'"))


def transcribe_words(audio: Path, cfg: dict, prompt: str = "") -> list[tuple[str, float, float]]:
    """Whisper's words with timestamps, as (word, start, end)."""
    model = _load(cfg)
    with _LOCK:
        segments, _ = model.transcribe(
            _pcm(audio), language="en", word_timestamps=True, beam_size=5,
            condition_on_previous_text=False, vad_filter=False,
            initial_prompt=prompt[:600] or None,
        )
        segments = list(segments)  # the generator runs the model; finish it under the lock
    words = []
    for segment in segments:
        for w in segment.words or []:
            text = w.word.strip()
            if text:
                words.append((text, float(w.start), float(w.end)))
    return words


def align_words(audio: Path, text: str, cfg: dict) -> list | None:
    """[word, start, end] for every word of ``text`` as spoken in ``audio``, or None.

    Script words Whisper did not recognise (numbers it spelled out, a misheard
    name) are spread between their recognised neighbours by letter count.
    """
    tokens = text.split()
    if not tokens:
        return []
    heard = transcribe_words(audio, cfg, prompt=text)
    if not heard:
        return None
    matcher = difflib.SequenceMatcher(None, [_norm(t) for t in tokens], [_norm(h[0]) for h in heard],
                                      autojunk=False)
    anchors: dict[int, tuple[float, float]] = {}
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            anchors[block.a + k] = (heard[block.b + k][1], heard[block.b + k][2])
    if len(anchors) < max(1, len(tokens) * MIN_MATCH_RATIO):
        log(f"captions: Whisper matched only {len(anchors)}/{len(tokens)} words; using estimated timing")
        return None

    clip_end = max(h[2] for h in heard)
    words: list = [None] * len(tokens)
    i = 0
    while i < len(tokens):
        if i in anchors:
            start, end = anchors[i]
            words[i] = [tokens[i], round(start, 3), round(max(end, start + 0.05), 3)]
            i += 1
            continue
        j = i
        while j < len(tokens) and j not in anchors:
            j += 1
        gap_start = anchors[i - 1][1] if i > 0 else (heard[0][1] if j == len(tokens) else 0.0)
        gap_end = anchors[j][0] if j < len(tokens) else clip_end
        if gap_end <= gap_start:
            gap_end = gap_start + 0.05 * (j - i)
        weights = [max(len(_norm(t)), 1) for t in tokens[i:j]]
        total, cursor = sum(weights), gap_start
        for k, weight in zip(range(i, j), weights):
            share = (gap_end - gap_start) * weight / total
            words[k] = [tokens[k], round(cursor, 3), round(max(cursor + share, cursor + 0.05), 3)]
            cursor += share
        i = j
    # Keep starts monotone so a late anchor never overlaps the word before it.
    for k in range(1, len(words)):
        if words[k][1] < words[k - 1][1]:
            words[k][1] = words[k - 1][1]
            words[k][2] = max(words[k][2], words[k][1] + 0.05)
    return words
