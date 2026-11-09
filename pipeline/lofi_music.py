"""Music for lofi videos, generated on this machine and free to use commercially.

Two engines:

``procedural``  A small composer and synthesizer written here in numpy. It
    picks a key, tempo and chord loop, voices a warm electric piano, a round
    bass, swung drums and a sparse pentatonic melody, then runs the mix through
    the lofi treatment (low-pass, tape wobble, vinyl crackle, side-chain
    ducking). No model, no download, no licence but this project's. Each
    track is seeded, so a video can be re-rendered to the same music.

``acestep``     ACE-Step 1.5, an MIT-licensed open text-to-music model whose
    generated audio may be used commercially. It is a separate install (its
    own Python environment and about 10 GB of weights) that exposes a local
    HTTP API; this module talks to that API and falls back to the procedural
    engine when the server is not running. See README, "Lofi music videos".

Rain ambience for the video comes from here too (``rain_ambience``).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests

from .util import log, run

SR = 44100

DEFAULTS = {
    "engine": "procedural",             # procedural | acestep
    "track_minutes": 2.5,               # length of each generated track
    "fallback_to_procedural": True,     # when the ACE-Step server is unreachable or fails
    "acestep_url": "http://127.0.0.1:8001",
    "acestep_steps": 8,                 # turbo model: 8 is the documented default
    "acestep_timeout_minutes": 20,      # per track, on a small GPU with CPU offload
    "acestep_prompt": "lofi hip hop, chill study beat, warm electric piano chords, soft dusty drums, "
                      "vinyl crackle, mellow, relaxed, instrumental, no vocals",
}


@dataclass
class Track:
    path: str
    title: str
    seconds: float
    bpm: int
    key: str
    engine: str

    def to_dict(self) -> dict:
        return asdict(self)


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in (cfg.get("music", {}) or {}).items() if k in DEFAULTS})
    return merged


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Track titles
# ---------------------------------------------------------------------------

_ADJ = ["Quiet", "Paper", "Late", "Amber", "Rainy", "Faded", "Slow", "Velvet", "Hollow", "Warm",
        "Dusty", "Blue", "Soft", "Lantern", "Window", "Midnight", "Morning", "Empty", "Golden", "Grey"]
_NOUN = ["Lanterns", "Pages", "Trains", "Rooftops", "Teacups", "Streets", "Windows", "Letters",
         "Umbrellas", "Hallways", "Rivers", "Notebooks", "Sidewalks", "Balconies", "Clouds",
         "Afternoons", "Bookshelves", "Corridors", "Kettles", "Stations"]


def track_titles(count: int, seed_text: str, given: list[str] | None = None) -> list[str]:
    """Titles for the tracklist: the ones given first, then seeded two-word names."""
    rng = random.Random(_seed(seed_text) ^ 0x5EED)
    titles = [str(t).strip() for t in (given or []) if str(t).strip()]
    seen = set(t.lower() for t in titles)
    tries = 0
    while len(titles) < count and tries < 500:
        tries += 1
        name = f"{rng.choice(_ADJ)} {rng.choice(_NOUN)}"
        if name.lower() not in seen:
            seen.add(name.lower())
            titles.append(name)
    return titles[:count]


# ---------------------------------------------------------------------------
# Procedural composer
# ---------------------------------------------------------------------------

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJ7, MIN7, DOM7, MIN7B5 = (0, 4, 7, 11), (0, 3, 7, 10), (0, 4, 7, 10), (0, 3, 6, 10)
# Chord loops as (semitones above the key root, chord intervals); four bars each.
PROGRESSIONS = {
    "major": [
        [(2, MIN7), (7, DOM7), (0, MAJ7), (9, MIN7)],      # ii V I vi
        [(5, MAJ7), (4, MIN7), (2, MIN7), (7, DOM7)],      # IV iii ii V
        [(0, MAJ7), (9, MIN7), (2, MIN7), (7, DOM7)],      # I vi ii V
        [(9, MIN7), (5, MAJ7), (0, MAJ7), (7, DOM7)],      # vi IV I V
    ],
    "minor": [
        [(0, MIN7), (5, MIN7), (8, MAJ7), (7, DOM7)],      # i iv VI V
        [(0, MIN7), (8, MAJ7), (3, MAJ7), (10, DOM7)],     # i VI III VII
        [(2, MIN7B5), (7, DOM7), (0, MIN7), (0, MIN7)],    # ii° V i i
        [(5, MIN7), (8, MAJ7), (0, MIN7), (7, DOM7)],      # iv VI i V
    ],
}
PENTATONIC = {"major": (0, 2, 4, 7, 9), "minor": (0, 3, 5, 7, 10)}


def _midi_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n: int, attack: float, decay_tau: float, release: float, hold: int) -> np.ndarray:
    """Attack, exponential decay, then a linear release starting at ``hold`` samples."""
    t = np.arange(n, dtype=np.float32) / SR
    env = np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay_tau)
    rel = np.ones(n, dtype=np.float32)
    tail = n - hold
    if tail > 0:
        rel[hold:] = np.linspace(1.0, 0.0, tail, dtype=np.float32)
    return (env * rel).astype(np.float32)


def _ep_note(midi: float, seconds: float, bright: float = 1.0) -> np.ndarray:
    """Electric-piano tine: FM with a fast-decaying index, a little second harmonic, chorus."""
    hold = int(seconds * SR)
    n = hold + int(0.5 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = _midi_hz(midi)
    index = (0.9 * bright) * np.exp(-t * 2.5)
    out = np.zeros(n, dtype=np.float32)
    for detune, gain in ((0.0, 0.7), (0.004, 0.3)):
        ph = 2 * np.pi * f * (1 + detune) * t
        out += gain * np.sin(ph + index * np.sin(ph))
    out += 0.12 * bright * np.sin(4 * np.pi * f * t) * np.exp(-t * 6)
    return out * _env(n, 0.006, 1.6, 0.4, hold) * 0.5


def _bass_note(midi: float, seconds: float) -> np.ndarray:
    hold = int(seconds * SR)
    n = hold + int(0.15 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    f = _midi_hz(midi)
    ph = 2 * np.pi * f * t
    tri = (2 / np.pi) * np.arcsin(np.sin(ph))
    out = 0.8 * np.sin(ph) + 0.25 * tri * np.exp(-t * 3)
    return out * _env(n, 0.008, 1.1, 0.1, hold) * 0.6


def _pad_chord(midis: list[float], seconds: float) -> np.ndarray:
    hold = int(seconds * SR)
    n = hold + int(1.0 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    out = np.zeros(n, dtype=np.float32)
    for m in midis:
        f = _midi_hz(m)
        for k in (1, 2, 3):
            out += np.sin(2 * np.pi * f * k * t + 0.3 * k) / (k * k)
        out += 0.5 * np.sin(2 * np.pi * f * 1.003 * t)
    env = np.minimum(t / 1.2, 1.0)
    rel = np.ones(n, dtype=np.float32)
    rel[hold:] = np.linspace(1.0, 0.0, n - hold, dtype=np.float32)
    return out * env * rel * (0.12 / max(len(midis), 1))


def _one_pole(x: np.ndarray, cutoff: float, highpass: bool = False) -> np.ndarray:
    """Tiny recursive filter for short drum samples (python loop; samples are short)."""
    a = math.exp(-2 * math.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = a * acc + (1 - a) * float(x[i])
        y[i] = acc
    return (x - y) if highpass else y


def _drum_kit(rng: random.Random) -> dict[str, np.ndarray]:
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    # Kick: pitch sweep with a soft click.
    n = int(0.45 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    freq = 42 + 85 * np.exp(-t * 22)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    kick = np.sin(phase) * np.exp(-t * 9) + 0.15 * nrng.standard_normal(n).astype(np.float32) * np.exp(-t * 180)
    kick = np.tanh(kick * 1.6) * 0.9
    # Snare: body tone plus band-limited noise.
    n = int(0.28 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    noise = nrng.standard_normal(n).astype(np.float32)
    noise = _one_pole(_one_pole(noise, 7000), 900, highpass=True)
    snare = 0.5 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 28) + 1.4 * noise * np.exp(-t * 18)
    snare = np.tanh(snare) * 0.55
    # Hats: short and long bursts of high noise.
    n = int(0.09 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    hat = _one_pole(nrng.standard_normal(n).astype(np.float32), 6500, highpass=True) * np.exp(-t * 55) * 0.35
    n = int(0.32 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    ohat = _one_pole(nrng.standard_normal(n).astype(np.float32), 6000, highpass=True) * np.exp(-t * 9) * 0.28
    # Rim: a woody tick for ghost notes.
    n = int(0.05 * SR)
    t = np.arange(n, dtype=np.float32) / SR
    rim = np.sin(2 * np.pi * 820 * t) * np.exp(-t * 90) * 0.4
    return {"kick": kick.astype(np.float32), "snare": snare.astype(np.float32),
            "hat": hat.astype(np.float32), "ohat": ohat.astype(np.float32), "rim": rim.astype(np.float32)}


def _add(buf: np.ndarray, start: int, sample: np.ndarray, gain: float = 1.0, pan: float = 0.0) -> None:
    """Mix a mono sample into a stereo buffer at ``start`` with constant-power panning."""
    if start >= len(buf):
        return
    if start < 0:
        sample = sample[-start:]
        start = 0
    end = min(len(buf), start + len(sample))
    if end <= start:
        return
    seg = sample[:end - start]
    angle = (pan + 1) * np.pi / 4
    buf[start:end, 0] += seg * gain * np.cos(angle)
    buf[start:end, 1] += seg * gain * np.sin(angle)


def _fft_filter(x: np.ndarray, cutoff: float, order: float = 2.0, highpass: bool = False) -> np.ndarray:
    """Butterworth-shaped magnitude filter applied in the frequency domain (whole-track, fast)."""
    n = x.shape[0]
    freqs = np.fft.rfftfreq(n, 1 / SR)
    ratio = freqs / max(cutoff, 1.0)
    with np.errstate(divide="ignore"):
        mag = 1 / np.sqrt(1 + ratio ** (2 * order)) if not highpass else \
            1 / np.sqrt(1 + np.where(ratio > 0, ratio, 1e-9) ** (-2 * order))
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = np.fft.irfft(np.fft.rfft(x[:, ch]) * mag, n=n)
    return out.astype(np.float32)


def _tape_wobble(x: np.ndarray, rng: random.Random, depth: float = 0.0022, rate: float = 0.45) -> np.ndarray:
    n = x.shape[0]
    i = np.arange(n, dtype=np.float64)
    amp = depth * SR / (2 * np.pi * rate)
    pos = i + amp * np.sin(2 * np.pi * rate * i / SR + rng.random() * 6.28) \
        + 0.35 * amp * np.sin(2 * np.pi * rate * 3.1 * i / SR)
    pos = np.clip(pos, 0, n - 1)
    out = np.empty_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = np.interp(pos, i, x[:, ch].astype(np.float64)).astype(np.float32)
    return out


def _vinyl(n: int, rng: random.Random, level: float = 0.012) -> np.ndarray:
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    crackle = np.zeros(n, dtype=np.float32)
    hits = nrng.random(n) < (22.0 / SR)
    crackle[hits] = nrng.standard_normal(int(hits.sum())).astype(np.float32) * 0.8
    kernel = np.exp(-np.arange(48) / 9.0).astype(np.float32)
    crackle = np.convolve(crackle, kernel, mode="same")
    hiss = nrng.standard_normal(n).astype(np.float32) * 0.08
    mono = (crackle + hiss) * level
    stereo = np.stack([mono, np.roll(mono, 37)], axis=1)
    return _fft_filter(stereo, 4500, order=1.5)


def rain_ambience(seconds: float, seed_text: str = "rain") -> np.ndarray:
    """Soft rain: filtered noise with a slow swell and scattered droplet ticks (stereo float32)."""
    rng = random.Random(_seed(seed_text))
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    n = int(seconds * SR)
    base = nrng.standard_normal((n, 2)).astype(np.float32)
    base = _fft_filter(base, 2600, order=1.2)
    base = _fft_filter(base, 180, order=1.5, highpass=True)
    t = np.arange(n, dtype=np.float32) / SR
    swell = 0.8 + 0.2 * np.sin(2 * np.pi * 0.05 * t) * np.sin(2 * np.pi * 0.013 * t + 1.0)
    base *= swell[:, None]
    drops = np.zeros((n, 2), dtype=np.float32)
    for ch in range(2):
        mask = nrng.random(n) < (9.0 / SR)
        drops[mask, ch] = nrng.standard_normal(int(mask.sum())).astype(np.float32)
    kernel = np.exp(-np.arange(220) / 40.0).astype(np.float32) * np.sin(np.arange(220) * 0.35).astype(np.float32)
    for ch in range(2):
        drops[:, ch] = np.convolve(drops[:, ch], kernel, mode="same")
    out = base * 0.18 + drops * 0.05
    peak = float(np.max(np.abs(out)) or 1.0)
    return (out / peak * 0.5).astype(np.float32)


def compose(seed_text: str, minutes: float) -> tuple[np.ndarray, dict]:
    """Render one lofi track as a stereo float32 array; returns (audio, info)."""
    rng = random.Random(_seed(seed_text))
    bpm = rng.randint(68, 88)
    mode = rng.choice(["major", "minor", "minor"])
    root = rng.randrange(12)
    progression = rng.choice(PROGRESSIONS[mode])
    beat = 60.0 / bpm
    bar = 4 * beat
    swing = rng.uniform(0.05, 0.11) * beat          # delay of off-beat eighths
    target_bars = max(16, int(round(minutes * 60 / bar)))
    # Sections: intro, A, B (melody), A', bridge (no drums), B', outro.
    plan = [("intro", 4), ("A", 8), ("B", 8), ("A", 8), ("bridge", 4), ("B", 8), ("outro", 4)]
    while sum(b for _, b in plan) < target_bars:
        plan.insert(-1, ("B", 8))
    total_bars = sum(b for _, b in plan)
    n = int(total_bars * bar * SR) + int(3 * SR)
    keys = np.zeros((n, 2), dtype=np.float32)
    bass = np.zeros((n, 2), dtype=np.float32)
    drums = np.zeros((n, 2), dtype=np.float32)
    lead = np.zeros((n, 2), dtype=np.float32)
    pad = np.zeros((n, 2), dtype=np.float32)
    kit = _drum_kit(rng)
    kick_times: list[float] = []
    pent = PENTATONIC[mode]

    def at(bar_index: int, step: int, jitter: float = 0.006) -> float:
        t = bar_index * bar + step * beat / 4
        if step % 4 == 2:
            t += swing
        return t + rng.uniform(-jitter, jitter)

    # A two-bar melodic motif reused with small variations keeps the melody memorable.
    motif = []
    for step in range(0, 32, 2):
        if rng.random() < 0.42:
            motif.append((step, rng.choice(pent) + 12 * rng.choice([0, 0, 1]), rng.choice([2, 2, 3, 4])))

    bar_index = 0
    for section, bars in plan:
        for b in range(bars):
            chord_root, intervals = progression[(bar_index) % len(progression)]
            root_midi = 48 + root + chord_root
            voicing = [root_midi + i for i in intervals]
            voicing[0] += 12 if rng.random() < 0.5 else 0
            voicing = sorted(voicing)
            melodic = section in ("B",) or (section == "A" and b >= 4 and rng.random() < 0.3)
            with_drums = section in ("A", "B") or (section == "outro" and b == 0)
            keys_gain = 0.75 if section != "intro" else 0.6
            if section == "outro":
                keys_gain *= max(0.15, 1 - b / bars)

            # Electric piano: a chord on the one, a re-voiced answer later in the bar.
            hits = [(0, 1.0)] + ([(10, 0.8)] if rng.random() < 0.8 else [(8, 0.7)])
            for step, vel in hits:
                top = voicing if step == 0 else [m + (12 if k == 0 else 0) for k, m in enumerate(voicing)]
                for k, midi in enumerate(top):
                    start = at(bar_index, step) + k * 0.014
                    _add(keys, int(start * SR), _ep_note(midi, beat * 1.8, bright=0.9),
                         gain=keys_gain * vel * rng.uniform(0.85, 1.0), pan=rng.uniform(-0.35, 0.35))
            # Bass: root on the one, a fifth or octave answer, sometimes an approach note.
            if section != "intro":
                low = 36 + root + chord_root
                _add(bass, int(at(bar_index, 0) * SR), _bass_note(low, beat * 1.6), gain=0.9)
                answer = low + rng.choice([7, 12, 7, 5])
                _add(bass, int(at(bar_index, 10) * SR), _bass_note(answer, beat * 0.7), gain=0.7)
                if rng.random() < 0.35:
                    nxt_root, _ = progression[(bar_index + 1) % len(progression)]
                    approach = 36 + root + nxt_root - rng.choice([1, 2])
                    _add(bass, int(at(bar_index, 14) * SR), _bass_note(approach, beat * 0.45), gain=0.55)
            # Drums with swing and ghost notes.
            if with_drums:
                kicks = [0, 10] + ([7] if rng.random() < 0.3 else []) + ([13] if rng.random() < 0.2 else [])
                for step in kicks:
                    t = at(bar_index, step)
                    kick_times.append(t)
                    _add(drums, int(t * SR), kit["kick"], gain=rng.uniform(0.85, 1.0))
                for step in (4, 12):
                    _add(drums, int(at(bar_index, step) * SR), kit["snare"], gain=rng.uniform(0.8, 1.0))
                if rng.random() < 0.45:
                    _add(drums, int(at(bar_index, rng.choice([7, 14, 15])) * SR), kit["rim"], gain=0.5)
                for step in range(0, 16, 2):
                    gain = 0.75 if step % 4 == 0 else 0.5
                    sample = kit["ohat"] if (step == 14 and rng.random() < 0.25) else kit["hat"]
                    _add(drums, int(at(bar_index, step) * SR), sample, gain=gain * rng.uniform(0.8, 1.0), pan=0.3)
            # Pad under the bridge.
            if section == "bridge":
                _add(pad, int(at(bar_index, 0) * SR), _pad_chord([m + 12 for m in voicing], bar * 1.05), gain=1.0)
            # Melody from the motif, kept inside the chord's key.
            if melodic:
                phase = b % 2
                for step, degree, length in motif:
                    if step // 16 != phase:
                        continue
                    local = step % 16
                    if rng.random() < 0.15:
                        continue
                    midi = 72 + root + degree
                    _add(lead, int(at(bar_index, local) * SR), _ep_note(midi, beat * length / 2, bright=1.2),
                         gain=0.42 * rng.uniform(0.8, 1.0), pan=rng.uniform(-0.2, 0.2))
            bar_index += 1

    # Buses: dark keys, a touch of width on the lead, drums slightly crushed.
    keys = _fft_filter(keys, 4200, order=1.5)
    lead = _fft_filter(lead, 5200, order=1.5)
    bass = _fft_filter(bass, 220, order=1.5)
    drums = np.tanh(drums * 1.4) * 0.8
    mix = keys + bass + lead + pad * 1.0
    # Side-chain: everything but the drums dips on each kick.
    duck = np.ones(n, dtype=np.float32)
    curve = 1 - 0.3 * np.exp(-np.arange(int(0.22 * SR), dtype=np.float32) / (0.09 * SR))
    for t in kick_times:
        s = int(t * SR)
        e = min(n, s + len(curve))
        if 0 <= s < n:
            duck[s:e] = np.minimum(duck[s:e], curve[:e - s])
    mix *= duck[:, None]
    mix += drums
    mix = _tape_wobble(mix, rng)
    mix = _fft_filter(mix, 9000, order=1.2)
    mix = _fft_filter(mix, 35, order=2, highpass=True)
    mix += _vinyl(n, rng)
    # Fades, soft clip, loudness.
    fade_in, fade_out = int(1.5 * SR), int(4 * SR)
    mix[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)[:, None]
    mix[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)[:, None]
    mix = np.tanh(mix * 1.3) / 1.3
    rms = float(np.sqrt(np.mean(mix ** 2)) or 1.0)
    mix *= min(0.16 / rms, 0.95 / float(np.max(np.abs(mix)) or 1.0))
    info = {"bpm": bpm, "key": f"{NOTE_NAMES[root]} {mode}", "bars": total_bars,
            "seconds": round(n / SR, 2)}
    return mix.astype(np.float32), info


def _write_mp3(audio: np.ndarray, mp3: Path) -> None:
    import soundfile as sf
    wav = mp3.with_suffix(".wav")
    sf.write(str(wav), audio, SR, subtype="PCM_16")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)])
    wav.unlink(missing_ok=True)


def _duration(path: Path) -> float:
    from .util import duration_of
    return duration_of(path)


def _procedural_track(out_dir: Path, index: int, title: str, seed_text: str, minutes: float) -> Track:
    mp3 = out_dir / f"track_{index + 1:02d}.mp3"
    meta = mp3.with_suffix(".json")
    if mp3.exists() and meta.exists():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved.get("seed_text") == seed_text and saved.get("engine") == "procedural":
            return Track(**{k: saved[k] for k in Track.__dataclass_fields__})
    audio, info = compose(seed_text, minutes)
    _write_mp3(audio, mp3)
    track = Track(path=mp3.name, title=title, seconds=round(_duration(mp3), 2),
                  bpm=info["bpm"], key=info["key"], engine="procedural")
    meta.write_text(json.dumps(dict(track.to_dict(), seed_text=seed_text), indent=2), encoding="utf-8")
    return track


# ---------------------------------------------------------------------------
# ACE-Step (local HTTP API)
# ---------------------------------------------------------------------------

def acestep_available(cfg: dict) -> bool:
    s = settings(cfg)
    try:
        return requests.get(f"{s['acestep_url'].rstrip('/')}/health", timeout=3).ok
    except requests.RequestException:
        return False


def acestep_hint(cfg: dict) -> str:
    return (f"ACE-Step's API server is not answering at {settings(cfg)['acestep_url']}. Install ACE-Step 1.5 "
            "in its own environment (git clone https://github.com/ACE-Step/ACE-Step-1.5, uv sync) and start "
            "`uv run acestep-api`, or set music.engine to procedural.")


def _acestep_track(cfg: dict, out_dir: Path, index: int, title: str, seed_text: str, minutes: float) -> Track:
    s = settings(cfg)
    base = s["acestep_url"].rstrip("/")
    mp3 = out_dir / f"track_{index + 1:02d}.mp3"
    meta = mp3.with_suffix(".json")
    if mp3.exists() and meta.exists():
        saved = json.loads(meta.read_text(encoding="utf-8"))
        if saved.get("seed_text") == seed_text and saved.get("engine") == "acestep":
            return Track(**{k: saved[k] for k in Track.__dataclass_fields__})
    seed = _seed(seed_text) % (2 ** 31)
    body = {
        "prompt": s["acestep_prompt"], "lyrics": "[Instrumental]",
        "audio_duration": int(max(10, min(600, minutes * 60))),
        "inference_steps": int(s["acestep_steps"]), "audio_format": "mp3",
        "seed": seed, "use_random_seed": False, "batch_size": 1, "thinking": False,
    }
    resp = requests.post(f"{base}/release_task", json=body, timeout=60)
    resp.raise_for_status()
    task_id = (resp.json().get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"ACE-Step did not return a task id: {resp.text[:200]}")
    deadline = time.time() + float(s["acestep_timeout_minutes"]) * 60
    file_url = None
    while time.time() < deadline:
        time.sleep(5)
        poll = requests.post(f"{base}/query_result", json={"task_id_list": [task_id]}, timeout=30)
        poll.raise_for_status()
        data = poll.json().get("data")
        entries = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        entry = next((e for e in entries if isinstance(e, dict)), None)
        if not entry:
            continue
        status = int(entry.get("status", 0) or 0)
        if status not in (1, 2):
            continue
        # ``result`` is a JSON string holding a list of items (one per generated file).
        result = entry.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except ValueError:
                result = []
        items = [x for x in (result if isinstance(result, list) else [result]) if isinstance(x, dict)]
        if status == 2:
            # The server reports failures in the item's ``error`` or only in ``progress_text``.
            reason = next((x.get("error") for x in items if x.get("error")), None) or entry.get("error")
            if not reason:
                text = str(entry.get("progress_text") or "")
                reason = next((ln for ln in reversed(text.splitlines()) if "Error" in ln or "error" in ln), text[:200])
            raise RuntimeError(f"ACE-Step failed: {str(reason)[:300]}")
        file_url = next((x["file"] for x in items if x.get("file")), None)
        if not file_url:
            raise RuntimeError("ACE-Step finished without an audio file")
        break
    if not file_url:
        raise RuntimeError("ACE-Step timed out before the track was ready")
    if file_url.startswith("/"):
        file_url = base + file_url
    audio = requests.get(file_url, timeout=120)
    audio.raise_for_status()
    mp3.write_bytes(audio.content)
    track = Track(path=mp3.name, title=title, seconds=round(_duration(mp3), 2), bpm=0, key="", engine="acestep")
    meta.write_text(json.dumps(dict(track.to_dict(), seed_text=seed_text), indent=2), encoding="utf-8")
    return track


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _kaggle_tracks(cfg: dict, out_dir: Path, names: list[str], seed_text: str, minutes: float) -> list[Track]:
    """Every missing track in one Kaggle job; tracks already on disk for this seed are kept."""
    from . import kaggle_music
    s = settings(cfg)
    tracks: list[Track | None] = []
    wanted = []
    for i, title in enumerate(names):
        mp3 = out_dir / f"track_{i + 1:02d}.mp3"
        meta = mp3.with_suffix(".json")
        seed = f"{seed_text}|{i}"
        if mp3.exists() and meta.exists():
            saved = json.loads(meta.read_text(encoding="utf-8"))
            if saved.get("seed_text") == seed and saved.get("engine") == "kaggle":
                tracks.append(Track(**{k: saved[k] for k in Track.__dataclass_fields__}))
                continue
        tracks.append(None)
        wanted.append({"id": mp3.stem, "title": title, "seed": _seed(seed) % (2 ** 31), "index": i})
    if wanted:
        log(f"music: {len(wanted)} tracks to generate on Kaggle (one job)")
        summary = kaggle_music.run_job(out_dir, wanted, int(max(10, min(600, minutes * 60))),
                                       s["acestep_prompt"], int(s["acestep_steps"]), cfg)
        for item in wanted:
            mp3 = out_dir / f"{item['id']}.mp3"
            if item["id"] not in summary["done"] or not mp3.exists():
                raise RuntimeError(f"Kaggle did not return {item['id']}: "
                                   f"{summary['failed'].get(item['id'], 'missing from the output')}")
            track = Track(path=mp3.name, title=item["title"], seconds=round(_duration(mp3), 2),
                          bpm=0, key="", engine="kaggle")
            mp3.with_suffix(".json").write_text(
                json.dumps(dict(track.to_dict(), seed_text=f"{seed_text}|{item['index']}"), indent=2), encoding="utf-8")
            tracks[item["index"]] = track
    return [t for t in tracks if t is not None]


def generate_tracks(cfg: dict, out_dir: Path, total_minutes: float, seed_text: str,
                    titles: list[str] | None = None, engine: str | None = None) -> list[Track]:
    """Enough tracks to fill ``total_minutes``; reuses tracks already on disk for the same seed."""
    s = settings(cfg)
    engine = engine or s["engine"]
    per = max(0.5, float(s["track_minutes"]))
    count = max(1, math.ceil(total_minutes / per))
    out_dir.mkdir(parents=True, exist_ok=True)
    names = track_titles(count, seed_text, titles)
    tracks: list[Track] = []
    if engine == "kaggle":
        try:
            return _kaggle_tracks(cfg, out_dir, names, seed_text, per)
        except Exception as exc:
            if not s["fallback_to_procedural"]:
                raise
            log(f"music: Kaggle failed; composing procedurally instead. {type(exc).__name__}: {str(exc)[:1500]}")
            engine = "procedural"
    if engine == "acestep" and not acestep_available(cfg):
        if not s["fallback_to_procedural"]:
            raise RuntimeError(acestep_hint(cfg))
        log("music: " + acestep_hint(cfg).split(".")[0] + "; composing procedurally instead")
        engine = "procedural"
    for i in range(count):
        seed = f"{seed_text}|{i}"
        log(f"music: track {i + 1}/{count} '{names[i]}' via {engine}")
        if engine == "acestep":
            try:
                tracks.append(_acestep_track(cfg, out_dir, i, names[i], seed, per))
                continue
            except Exception as exc:
                if not s["fallback_to_procedural"]:
                    raise
                hint = ""
                if "out of memory" in str(exc).lower():
                    hint = " (the GPU is full: close other GPU users, such as the studio's AI panel, before rendering)"
                log(f"music: ACE-Step failed ({type(exc).__name__}: {str(exc)[:160]}){hint}; composing this one procedurally")
        try:
            tracks.append(_procedural_track(out_dir, i, names[i], seed, per))
        except MemoryError as exc:
            raise RuntimeError("Out of system memory while composing a track. Close other programs; ACE-Step's "
                               "server with CPU offload alone holds about 10 GB of RAM.") from exc
    return tracks
