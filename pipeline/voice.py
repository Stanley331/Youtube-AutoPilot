"""Stage 3: narration audio per scene, with word timestamps for captions.

Engines:
  kokoro      - local ONNX neural voice; no API key; word timing comes from the model's phoneme timings
  gemini      - Gemini Flash TTS on the free tier (GEMINI_API_KEY); timing is estimated
  elevenlabs  - natural voice, returns character-level timing we turn into word timing
  local       - free built-in voice for testing (macOS "say" or Windows SAPI); timing is estimated

Which narrator speaks is decided in pipeline/voices.py; this module only speaks.
"""
import base64
import hashlib
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .config import ROOT, env
from .fonts import is_windows
from .util import duration_of, log, run, write_json, read_json


@dataclass
class SceneAudio:
    path: str
    duration: float
    words: list = field(default_factory=list)  # [[word, start, end], ...] relative to scene
    engine: str = ""
    voice: str = ""
    timing: str = ""  # "model" (engine reported it), "whisper" (aligned locally), "estimated"


_KOKORO = None
_KOKORO_LOCK = threading.RLock()  # one ONNX session, built once, used by one thread at a time
KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/kokoro-v1.0.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.1/voices-v1.0.bin"
)

GEMINI_TTS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
DEFAULT_GEMINI_TTS_MODEL = "gemini-3.8-flash-tts"
DEFAULT_GEMINI_TTS_FALLBACK = "gemini-3.8-flash-lite-tts"
GEMINI_TTS_RETRY_STATUSES = {429, 500, 502, 503, 504}

PREVIEW_TEXT = "Every story has a door. This one opened at midnight, and nobody heard it close."


def narration_digest(text: str) -> str:
    """Stable fingerprint of the narration so edited scripts get fresh audio."""
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def load_scene_audio(meta: Path) -> SceneAudio | None:
    """Rebuild a SceneAudio from the JSON written next to each mp3."""
    if not meta.exists():
        return None
    try:
        saved = read_json(meta)
    except (OSError, ValueError):
        return None
    if not isinstance(saved, dict) or "duration" not in saved:
        return None
    return SceneAudio(
        path=saved.get("path", meta.with_suffix(".mp3").name),
        duration=float(saved["duration"]),
        words=saved.get("words", []),
        engine=saved.get("engine", ""),
        voice=saved.get("voice", ""),
        timing=saved.get("timing", ""),
    )


def synthesize_all(scenes: list[dict], out_dir: Path, cfg: dict, engine: str,
                   voice: str = "") -> list[SceneAudio]:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    results = []
    for i, scene in enumerate(scenes):
        narration = scene.get("narration", "")
        mp3 = audio_dir / f"scene_{i:02d}.mp3"
        meta = audio_dir / f"scene_{i:02d}.json"
        digest = narration_digest(narration)
        if mp3.exists() and meta.exists():
            saved = read_json(meta)
            # Reuse only when the same engine and narrator spoke the same words. Older
            # metadata has no digest or voice; treat it as current so existing projects keep working.
            if (saved.get("engine") == engine
                    and saved.get("voice", voice) == voice
                    and saved.get("narration_digest", digest) == digest
                    and (audio_dir / saved.get("path", mp3.name)).exists()):
                results.append(SceneAudio(
                    path=saved["path"],
                    duration=saved["duration"],
                    words=saved["words"],
                    engine=engine,
                    voice=voice,
                    timing=saved.get("timing", ""),
                ))
                continue
        log(f"voice: scene {i + 1}/{len(scenes)} via {engine}" + (f" ({voice})" if voice else ""))
        audio = speak(narration, mp3, cfg, engine, voice)
        # A fresh take replaces any speed-adjusted copy of an older narration.
        (audio_dir / f"scene_{i:02d}.orig.mp3").unlink(missing_ok=True)
        record = dict(audio.__dict__)
        record["narration_digest"] = digest
        write_json(meta, record)
        results.append(audio)
    return results


def speak(text: str, mp3: Path, cfg: dict, engine: str, voice: str = "") -> SceneAudio:
    """Synthesize one passage with the given engine and narrator."""
    if engine == "elevenlabs":
        audio = _elevenlabs(text, mp3, cfg, voice)
    elif engine == "kokoro":
        audio = _kokoro(text, mp3, cfg, voice)
    elif engine == "gemini":
        audio = _gemini(text, mp3, cfg, voice)
    elif engine == "local":
        audio = _local(text, mp3, cfg, voice)
    else:
        raise ValueError(f"unknown voice engine '{engine}'")
    audio.engine = engine
    audio.voice = voice
    return audio


def preview_voice(cfg: dict, engine: str, voice: str = "", text: str = PREVIEW_TEXT) -> Path:
    """A short sample of a narrator, cached under cache/tts/previews for the UI."""
    preview_dir = ROOT / "cache" / "tts" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    mp3 = preview_dir / f"{engine}_{voice or 'default'}_{tag}.mp3"
    if not mp3.exists():
        speak(text, mp3, cfg, engine, voice)
    return mp3


def apply_scene_speed(out_dir: Path, index: int, speed: float) -> SceneAudio:
    """Re-time one scene's narration (0.5x to 2.0x) and update its metadata.

    The first adjustment keeps a copy of the original take, so later changes are
    always derived from the untouched audio instead of stacking resamples.
    """
    speed = float(speed)
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be between 0.5 and 2.0")
    audio_dir = out_dir / "audio"
    mp3 = audio_dir / f"scene_{index:02d}.mp3"
    meta = audio_dir / f"scene_{index:02d}.json"
    orig = audio_dir / f"scene_{index:02d}.orig.mp3"
    if not mp3.exists() or not meta.exists():
        raise RuntimeError(f"scene {index + 1} has no narration yet")
    saved = read_json(meta)
    if not orig.exists():
        shutil.copyfile(mp3, orig)
        saved["orig_duration"] = saved["duration"]
        saved["orig_words"] = saved.get("words", [])
    orig_duration = float(saved.get("orig_duration", saved["duration"]))
    orig_words = saved.get("orig_words", saved.get("words", []))

    if abs(speed - 1.0) < 1e-6:
        shutil.copyfile(orig, mp3)
        new_duration = duration_of(mp3)
    else:
        tmp = mp3.with_name(f"scene_{index:02d}.speed.mp3")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(orig),
             "-filter:a", f"atempo={speed:.4f}", "-codec:a", "libmp3lame", "-q:a", "2", str(tmp)])
        tmp.replace(mp3)
        new_duration = duration_of(mp3)
    ratio = new_duration / orig_duration if orig_duration else 1.0
    words = [[w, round(s * ratio, 3), round(e * ratio, 3)] for w, s, e in orig_words]
    saved.update({"duration": new_duration, "words": words, "speed": speed, "path": mp3.name})
    write_json(meta, saved)
    return SceneAudio(path=mp3.name, duration=new_duration, words=words,
                      engine=saved.get("engine", ""), voice=saved.get("voice", ""))


# ---------------------------------------------------------------------------
# ElevenLabs
# ---------------------------------------------------------------------------

def _elevenlabs(text: str, mp3: Path, cfg: dict, voice: str = "") -> SceneAudio:
    key = env("ELEVENLABS_API_KEY")
    voice_id = voice or env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # "Rachel", a default voice
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set; use --voice kokoro for local narration")
    v = cfg["voice"]
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
        params={"output_format": "mp3_44100_128"},
        headers={"xi-api-key": key, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": v["elevenlabs_model"],
            "voice_settings": {"stability": v["stability"], "similarity_boost": v["similarity_boost"]},
        },
        timeout=120,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    mp3.write_bytes(base64.b64decode(data["audio_base64"]))
    alignment = data.get("alignment") or data.get("normalized_alignment")
    if alignment:
        words = _words_from_alignment(alignment)
        return SceneAudio(path=mp3.name, duration=duration_of(mp3), words=words, timing="model")
    return _timed_audio(text, mp3, cfg)


def _words_from_alignment(al: dict) -> list:
    """Collapse ElevenLabs character timings into [word, start, end] triples."""
    chars, starts, ends = al["characters"], al["character_start_times_seconds"], al["character_end_times_seconds"]
    words, current, w_start, w_end = [], "", None, None
    for ch, s, e in zip(chars, starts, ends):
        if ch.isspace():
            if current:
                words.append([current, round(w_start, 3), round(w_end, 3)])
            current, w_start = "", None
        else:
            if not current:
                w_start = s
            current += ch
            w_end = e
    if current:
        words.append([current, round(w_start, 3), round(w_end, 3)])
    return words


# ---------------------------------------------------------------------------
# Kokoro (local)
# ---------------------------------------------------------------------------

def _kokoro(text: str, mp3: Path, cfg: dict, voice: str = "") -> SceneAudio:
    """Generate local narration with the Apache-licensed Kokoro ONNX model."""
    global _KOKORO
    with _KOKORO_LOCK:
        if _KOKORO is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:
                raise RuntimeError(
                    "Kokoro is not installed. Run: .\\.venv\\Scripts\\python.exe -m pip "
                    "install -r requirements.txt"
                ) from exc
            model_path, voices_path = _ensure_kokoro_models(cfg)
            _KOKORO = Kokoro(str(model_path), str(voices_path))

        voice = voice or cfg["voice"].get("kokoro_voice", "af_heart")
        speed = float(cfg["voice"].get("kokoro_speed", 1.0))
        language = cfg["voice"].get("kokoro_language", "en-us")
        timings = []
        if getattr(_KOKORO, "has_timings", False) and hasattr(_KOKORO, "create_timed"):
            samples, sample_rate, timings = _KOKORO.create_timed(text, voice=voice, speed=speed, lang=language)
        else:
            samples, sample_rate = _KOKORO.create(text, voice=voice, speed=speed, lang=language)

    import soundfile as sf

    wav = mp3.with_suffix(".wav")
    try:
        sf.write(str(wav), samples, sample_rate, subtype="PCM_16")
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
             "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)])
    finally:
        wav.unlink(missing_ok=True)
    words = _words_from_phonemes(text, timings) if timings else []
    if words:
        return SceneAudio(path=mp3.name, duration=duration_of(mp3), words=words, timing="model")
    return _timed_audio(text, mp3, cfg)


def _words_from_phonemes(text: str, timings: list) -> list:
    """Turn Kokoro's phoneme timings into [word, start, end] triples.

    The model separates words with a space phoneme. When the number of spoken
    groups matches the narration's words they map one to one; otherwise each
    word takes its share of the groups, which still follows the real pauses.
    """
    groups, current = [], []
    for t in timings:
        phoneme = getattr(t, "phoneme", "")
        if phoneme.isspace():
            if current:
                groups.append(_group(current))
                current = []
        elif phoneme.strip():
            current.append(t)
    if current:
        groups.append(_group(current))
    tokens = text.split()
    if not groups or not tokens:
        return []
    if len(groups) == len(tokens):
        return [[w, round(s, 3), round(max(e, s + 0.05), 3)] for w, (s, e, _, _) in zip(tokens, groups)]
    # espeak merges or splits a few words ("on the" -> one group, "1631" -> three).
    # Punctuation survives on both sides, so align run by run between marks.
    token_runs = _runs(tokens, lambda w: w)
    group_runs = _runs(groups, lambda g: g[3])
    if len(token_runs) == len(group_runs):
        words = []
        for t_run, g_run in zip(token_runs, group_runs):
            words += _spread(t_run, g_run)
        return words
    return _spread(tokens, groups)


_PUNCT = ".,;:!?"


def _group(phonemes: list) -> tuple:
    """(start, end, letter count, text) for one spoken word."""
    text = "".join(getattr(p, "phoneme", "") for p in phonemes)
    letters = sum(1 for ch in text if ch not in _PUNCT and not ch.isspace())
    return (phonemes[0].start, phonemes[-1].end, max(letters, 1), text)


def _runs(items: list, text_of) -> list[list]:
    runs, current = [], []
    for item in items:
        current.append(item)
        if text_of(item).rstrip()[-1:] in tuple(_PUNCT):
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _spread(tokens: list[str], groups: list[tuple]) -> list:
    """Map words onto spoken groups by letter share, so merged groups split their time."""
    weights = [max(sum(1 for ch in w if ch not in _PUNCT), 1) for w in tokens]
    total_t = sum(weights)
    total_g = sum(g[2] for g in groups)
    bounds, acc = [], 0.0
    for g in groups:
        bounds.append((acc, acc + g[2], g[0], g[1]))
        acc += g[2]

    def at(pos: float, closing: bool) -> float:
        for c0, c1, start, end in bounds:
            if (c0 < pos <= c1) if closing else (c0 <= pos < c1):
                return start + (pos - c0) / (c1 - c0) * (end - start)
        return bounds[-1][3] if closing else bounds[0][2]

    words, cursor = [], 0.0
    for word, weight in zip(tokens, weights):
        a = cursor / total_t * total_g
        b = (cursor + weight) / total_t * total_g
        start, end = at(a, False), at(b, True)
        words.append([word, round(start, 3), round(max(end, start + 0.05), 3)])
        cursor += weight
    return words


def _ensure_kokoro_models(cfg: dict) -> tuple[Path, Path]:
    model_dir = ROOT / cfg["voice"].get("kokoro_model_dir", "cache/tts/kokoro")
    model_dir.mkdir(parents=True, exist_ok=True)
    assets = [
        (model_dir / "kokoro-v1.0.onnx", KOKORO_MODEL_URL, 1_000_000),
        (model_dir / "voices-v1.0.bin", KOKORO_VOICES_URL, 100_000),
    ]
    for path, url, minimum_size in assets:
        if path.exists() and path.stat().st_size >= minimum_size:
            continue
        log(f"voice: downloading Kokoro model asset {path.name} (one-time setup)")
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                expected_size = response.headers.get("Content-Length")
                total = 0
                with open(partial, "wb") as output:
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            output.write(chunk)
                            total += len(chunk)
            if total < minimum_size or (expected_size and total != int(expected_size)):
                raise RuntimeError(f"downloaded model file {path.name} is incomplete")
            partial.replace(path)
        except (requests.RequestException, OSError, ValueError, RuntimeError) as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not set up Kokoro model file {path.name}: {type(exc).__name__}. "
                "Check your internet connection and run the voice stage again."
            ) from exc
    return assets[0][0], assets[1][0]


# ---------------------------------------------------------------------------
# Gemini Flash TTS (free tier)
# ---------------------------------------------------------------------------

def _gemini(text: str, mp3: Path, cfg: dict, voice: str = "") -> SceneAudio:
    """Speak through Gemini's TTS models; the response is a base64 WAV."""
    key = env("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set; use --voice kokoro for local narration")
    v = cfg["voice"]
    voice = voice or v.get("gemini_voice", "Sulafat")
    models = [env("GEMINI_TTS_MODEL", v.get("gemini_tts_model", DEFAULT_GEMINI_TTS_MODEL)),
              env("GEMINI_TTS_FALLBACK_MODEL", v.get("gemini_tts_fallback_model", DEFAULT_GEMINI_TTS_FALLBACK))]
    models = [m for i, m in enumerate(models) if m and m not in models[:i]]
    style = str(v.get("gemini_style", "") or "").strip()
    content = {"type": "text", "text": text}
    if style:
        content["annotations"] = [{"type": "speech_metadata", "style": style}]
    body = {
        "input": [{"type": "user_input", "content": [content]}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": [{"voice": voice}]},
    }
    attempts = int(v.get("gemini_retries", 4))
    last_error = "no response"
    for model in models:
        body["model"] = model
        for attempt in range(attempts):
            try:
                resp = requests.post(GEMINI_TTS_URL, json=body, timeout=(15, 120),
                                     headers={"x-goog-api-key": key, "Content-Type": "application/json"})
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}"
                time.sleep(5 * (attempt + 1))
                continue
            if resp.ok:
                data = _gemini_audio(resp.json())
                if data is None:
                    raise RuntimeError("Gemini TTS returned no audio for this passage")
                wav = mp3.with_suffix(".wav")
                try:
                    wav.write_bytes(data)
                    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                         "-codec:a", "libmp3lame", "-q:a", "2", str(mp3)])
                finally:
                    wav.unlink(missing_ok=True)
                return _timed_audio(text, mp3, cfg)
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code not in GEMINI_TTS_RETRY_STATUSES:
                break
            if resp.status_code == 429 and attempt + 1 == attempts:
                break  # quota: try the fallback model rather than waiting more
            wait = 20 * (attempt + 1) if resp.status_code == 429 else 5 * (attempt + 1)
            log(f"voice: Gemini TTS {resp.status_code}; retrying in {wait}s")
            time.sleep(wait)
        log(f"voice: Gemini TTS model {model} failed ({last_error[:80]})")
    raise RuntimeError(f"Gemini TTS failed: {last_error}. Switch to --voice kokoro to keep going offline.")


def _gemini_audio(payload: dict) -> bytes | None:
    for step in payload.get("steps", []) or []:
        for part in step.get("content", []) or []:
            if part.get("type") == "audio" and part.get("data"):
                return base64.b64decode(part["data"])
    return None


# ---------------------------------------------------------------------------
# Shared helpers and the system voice
# ---------------------------------------------------------------------------

def _timed_audio(text: str, mp3: Path, cfg: dict) -> SceneAudio:
    """Word timing for engines without alignment data: Whisper locally, else an even spread."""
    from . import align
    if align.settings(cfg)["align"]:
        if not align.available():
            log("captions: " + align.install_hint().replace("\n", " "))
        else:
            try:
                words = align.align_words(mp3, text, cfg)
            except Exception as exc:  # a broken model download must not sink the video
                log(f"captions: Whisper alignment failed ({type(exc).__name__}: {str(exc)[:120]})")
                words = None
            if words:
                return SceneAudio(path=mp3.name, duration=duration_of(mp3), words=words, timing="whisper")
    return _estimated_audio(text, mp3)


def _estimated_audio(text: str, mp3: Path) -> SceneAudio:
    """Spread the words evenly across the clip; the fallback when nothing better exists."""
    dur = duration_of(mp3)
    tokens = text.split()
    per = dur / max(len(tokens), 1)
    words = [[w, round(i * per, 3), round((i + 1) * per, 3)] for i, w in enumerate(tokens)]
    return SceneAudio(path=mp3.name, duration=dur, words=words, timing="estimated")


def _local(text: str, mp3: Path, cfg: dict, voice: str = "") -> SceneAudio:
    """Free built-in voice. Word timing is spread evenly, good enough for testing."""
    raw = mp3.with_suffix(".wav")
    if is_windows():
        _windows_sapi(text, raw, voice or cfg["voice"].get("windows_voice", ""))
    else:
        raw = mp3.with_suffix(".aiff")
        run(["say", "-v", voice or cfg["voice"].get("say_voice", "Samantha"), "-o", str(raw), text])
    try:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-codec:a", "libmp3lame",
             "-q:a", "2", str(mp3)])
    finally:
        raw.unlink(missing_ok=True)
    return _timed_audio(text, mp3, cfg)


def _windows_sapi(text: str, wav: Path, voice_name: str) -> None:
    """Speak with the Windows built-in synthesizer through PowerShell."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        text_file = Path(f.name)
    q = lambda p: str(p).replace("'", "''")
    select = f"try {{ $s.SelectVoice('{q(voice_name)}') }} catch {{}}" if voice_name else ""
    script = f"""
Add-Type -AssemblyName System.Speech
$t = Get-Content -Raw -Encoding UTF8 '{q(text_file)}'
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
{select}
$s.SetOutputToWaveFile('{q(wav)}')
$s.Speak($t)
$s.Dispose()
"""
    try:
        run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script])
    finally:
        text_file.unlink(missing_ok=True)
