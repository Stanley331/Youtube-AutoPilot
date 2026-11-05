"""Pixabay for lofi videos: an animated loop from its video API, and the music and
rain libraries you fill from its website, with a ledger so nothing repeats.

What the API gives us: ``https://pixabay.com/api/videos/`` with ``video_type=animation``
returns exactly the lofi loops channels use (a girl studying by a lamp, a cat
asleep by a rainy window), 8 to 76 seconds, landscape, under the Pixabay Content
License (free, commercial use allowed, no attribution required). A clip is made
seamless by cross-fading its tail into its own head, then repeated by stream copy.

What the API does not give us: music and sound effects. Pixabay serves those only
through its website, and its terms forbid automated downloads, so you download the
tracks you like from https://pixabay.com/music/ (search "lofi") and the rain from
https://pixabay.com/sound-effects/ into the two library folders in ``assets/``.
Keep the licence certificate the site offers next to each download; Pixabay music
is free to use in monetised videos, but some tracks are registered in YouTube's
Content ID and a claim is cleared by disputing it with that certificate.

``lofi_usage.json`` next to history.json records which clip, tracks and rain file
each video used; picks always prefer what has been used least.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import requests

from .config import ROOT, env
from .lofi_music import Track
from .util import duration_of, log, read_json, run, write_json

LEDGER = ROOT / "lofi_usage.json"
META_CACHE = ROOT / "cache" / "lofi_library_meta.json"
PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"
SEARCH_CACHE_SECONDS = 24 * 60 * 60
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}

DEFAULTS = {
    "music_dir": "assets/lofi_music",       # tracks downloaded from pixabay.com/music
    "ambience_dir": "assets/ambience",      # rain and other loops from pixabay.com/sound-effects
    "clip_cache_dir": "cache/clips",
    "clip_min_seconds": 5,
    "clip_max_seconds": 120,
    "clip_queries": [],                     # extra search terms; the scene's own come first
}

MUSIC_URL = "https://pixabay.com/music/search/lofi/"
AMBIENCE_URL = "https://pixabay.com/sound-effects/search/rain/"


def settings(cfg: dict) -> dict:
    merged = dict(DEFAULTS)
    lofi_cfg = cfg.get("lofi", {}) or {}
    music_cfg = cfg.get("music", {}) or {}
    merged.update({k: v for k, v in lofi_cfg.items() if k in DEFAULTS})
    if music_cfg.get("library_dir"):
        merged["music_dir"] = music_cfg["library_dir"]
    return merged


# ---------------------------------------------------------------------------
# The usage ledger
# ---------------------------------------------------------------------------

def ledger() -> dict:
    if LEDGER.exists():
        try:
            data = read_json(LEDGER)
            if isinstance(data, dict):
                return data
        except (OSError, ValueError):
            pass
    return {}


def uses(kind: str, key: str) -> int:
    return len((ledger().get(kind) or {}).get(str(key), []))


def record(kind: str, key: str, slug: str) -> None:
    """Note that ``slug`` used ``key`` (a clip id, a track file name, an ambience file name)."""
    data = ledger()
    bucket = data.setdefault(kind, {})
    used_by = bucket.setdefault(str(key), [])
    if slug not in used_by:
        used_by.append(slug)
    write_json(LEDGER, data)


# ---------------------------------------------------------------------------
# Music and ambience libraries (folders you fill from the website)
# ---------------------------------------------------------------------------

def _pretty_title(name: str) -> str:
    """'lofi-chill-310212.mp3' -> 'Lofi Chill'; Pixabay appends the numeric id to downloads."""
    stem = Path(name).stem
    stem = re.sub(r"[-_ ]?\d{4,}$", "", stem)
    stem = re.sub(r"[-_]+", " ", stem).strip()
    return " ".join(w.capitalize() for w in stem.split()) or Path(name).stem


def _meta_cache() -> dict:
    if META_CACHE.exists():
        try:
            return read_json(META_CACHE)
        except (OSError, ValueError):
            pass
    return {}


def _seconds_of(path: Path, cache: dict) -> float:
    key = f"{path.resolve()}|{path.stat().st_size}|{int(path.stat().st_mtime)}"
    if key in cache:
        return float(cache[key])
    try:
        seconds = duration_of(path)
    except Exception:
        seconds = 0.0
    cache[key] = seconds
    return seconds


def _audio_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES)


def library_tracks(cfg: dict) -> list[dict]:
    """Every track in the music library: name, title, seconds, how many videos used it."""
    s = settings(cfg)
    folder = ROOT / s["music_dir"]
    cache = _meta_cache()
    tracks = []
    for path in _audio_files(folder):
        seconds = _seconds_of(path, cache)
        if seconds <= 0:
            continue
        tracks.append({"path": str(path), "name": path.name, "title": _pretty_title(path.name),
                       "seconds": round(seconds, 2), "uses": uses("tracks", path.name)})
    META_CACHE.parent.mkdir(parents=True, exist_ok=True)
    write_json(META_CACHE, cache)
    return tracks


def library_status(cfg: dict) -> dict:
    s = settings(cfg)
    tracks = library_tracks(cfg)
    unused = [t for t in tracks if t["uses"] == 0]
    return {
        "dir": str(ROOT / s["music_dir"]), "count": len(tracks),
        "minutes": round(sum(t["seconds"] for t in tracks) / 60, 1),
        "unused": len(unused), "unused_minutes": round(sum(t["seconds"] for t in unused) / 60, 1),
        "ambience_dir": str(ROOT / s["ambience_dir"]),
        "ambience_files": len(_audio_files(ROOT / s["ambience_dir"])),
        "url": MUSIC_URL, "ambience_url": AMBIENCE_URL,
    }


def _spread(items: list[dict], seed_text: str) -> list[dict]:
    """Least used first; ties broken by a per-video hash so two videos do not share an order."""
    def key(item):
        h = hashlib.sha256(f"{seed_text}|{item['name']}".encode("utf-8")).hexdigest()
        return (item["uses"], h)
    return sorted(items, key=key)


def pick_tracks(cfg: dict, minutes: float, seed_text: str) -> list[Track]:
    """Enough library tracks for ``minutes``, least-used first, none twice in one video."""
    tracks = library_tracks(cfg)
    if not tracks:
        s = settings(cfg)
        raise RuntimeError(f"The music library is empty. Download lofi tracks from {MUSIC_URL} into "
                           f"{ROOT / s['music_dir']} (mp3 is fine), then try again.")
    chosen: list[Track] = []
    total = 0.0
    for item in _spread(tracks, seed_text):
        if total >= minutes * 60:
            break
        chosen.append(Track(path=item["path"], title=item["title"], seconds=item["seconds"],
                            bpm=0, key="", engine="pixabay"))
        total += item["seconds"]
    if total < minutes * 60:
        log(f"music: the library holds {total / 60:.0f} minutes, less than the {minutes:.0f} requested; "
            "the video will be as long as the music")
    return chosen


def record_tracks(tracks: list[Track], slug: str) -> None:
    for track in tracks:
        if track.engine == "pixabay":
            record("tracks", Path(track.path).name, slug)


def pick_ambience(cfg: dict, seed_text: str) -> Path | None:
    """The least-used rain file from the ambience library, or None when it is empty."""
    s = settings(cfg)
    files = _audio_files(ROOT / s["ambience_dir"])
    if not files:
        return None
    items = [{"name": p.name, "path": p, "uses": uses("ambience", p.name)} for p in files]
    return _spread(items, seed_text)[0]["path"]


# ---------------------------------------------------------------------------
# Animated loop clips from the video API
# ---------------------------------------------------------------------------

_STOP = {"a", "an", "the", "of", "at", "by", "in", "on", "with", "and", "to", "near", "under", "over",
         "from", "his", "her", "its", "their", "is", "are", "late", "night", "day", "warm"}
# Tags that say "this is a lofi scene" weigh more than tags that merely share a noun with it.
MOOD_WORDS = {"lofi", "lo-fi", "cozy", "chill", "relax", "relaxing", "study", "studying", "ambience",
              "ambient", "calm", "rain", "rainy", "night", "sleep", "sleeping", "asleep", "lamp", "fireplace",
              "window", "cafe", "coffee", "reading", "loop"}
GENRE_WORDS = MOOD_WORDS | {"anime", "cat", "train", "room", "girl", "cartoon", "animated", "cute"}


def scene_words(planned: dict) -> list[str]:
    text = " ".join(str(planned.get(k, "")) for k in ("visual_query", "visual_subject", "setting", "visual_description"))
    words = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w not in _STOP and len(w) > 2 and w not in words:
            words.append(w)
    return words


def relevance(candidate: dict, planned: dict) -> int:
    """How much a clip's tags look like this scene and like lofi at all."""
    tags = {t.strip() for t in str(candidate.get("tags", "")).lower().replace(",", " ").split()}
    scene = set(scene_words(planned)[:12])
    score = 3 * len(tags & scene) + 3 * len(tags & MOOD_WORDS) + len(tags & (GENRE_WORDS - MOOD_WORDS))
    if "lofi" in tags or "lo-fi" in tags:
        score += 8
    if planned.get("rainy") and ("rain" in tags or "rainy" in tags):
        score += 3
    # A five-second cartoon loops conspicuously; a minute-long scene breathes.
    duration = float(candidate.get("duration", 0) or 0)
    score += int(min(duration, 60) / 10)
    if duration < 15:
        score -= 6
    return score


def clip_queries(planned: dict, cfg: dict) -> list[str]:
    """Search terms for the loop: the scene's own first, then the genre's staples."""
    s = settings(cfg)
    base = " ".join(scene_words({"visual_query": planned.get("visual_query", "")})[:4])
    if len(base.split()) < 2:
        base = " ".join(scene_words(planned)[:4])
    queries = []
    if base:
        queries.append(f"{base} animation")
        queries.append(f"lofi {base}")
    if planned.get("rainy"):
        queries.append("lofi rain window animation")
    queries += [str(q) for q in s["clip_queries"] if str(q).strip()]
    queries += ["lofi animation loop", "lofi anime cozy animation"]
    seen, out = set(), []
    for q in queries:
        q = " ".join(q.split())[:100]
        if q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out


def _search(query: str, key: str, cache: Path, page: int = 1) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    cache_id = hashlib.sha256(f"{key}\0animation\0{query}\0{page}".encode()).hexdigest()
    cache_path = cache / f"loopsearch_{cache_id}.json"
    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < SEARCH_CACHE_SECONDS:
        try:
            return read_json(cache_path).get("hits", [])
        except (OSError, ValueError):
            pass
    try:
        resp = requests.get(PIXABAY_VIDEO_URL, params={
            "key": key, "q": query, "video_type": "animation", "orientation": "horizontal",
            "safesearch": "true", "order": "popular", "per_page": 40, "page": page,
        }, timeout=30)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    except (requests.RequestException, ValueError) as exc:
        log(f"pixabay: loop search failed for '{query}' ({type(exc).__name__})")
        return []
    write_json(cache_path, {"hits": hits})
    return hits


def _best_file(hit: dict, max_width: int = 1920) -> dict | None:
    files = hit.get("videos") or {}
    usable = [f for f in files.values()
              if isinstance(f, dict) and f.get("url") and f.get("width") and f.get("width") <= max_width
              and f.get("width", 0) >= f.get("height", 0)]
    if not usable:
        usable = [f for f in files.values() if isinstance(f, dict) and f.get("url") and f.get("width")]
    return max(usable, key=lambda f: f["width"], default=None)


def search_clips(cfg: dict, planned: dict, count: int = 12) -> list[dict]:
    """Candidate loops for the scene: landscape animations of a sensible length, least-used first."""
    key = env("PIXABAY_API_KEY")
    if not key:
        raise RuntimeError("PIXABAY_API_KEY is not set in .env; it is needed to search Pixabay for a loop")
    s = settings(cfg)
    cache = ROOT / s["clip_cache_dir"]
    seen: set[int] = set()
    found: list[dict] = []
    for query in clip_queries(planned, cfg):
        for hit in _search(query, key, cache):
            try:
                hid, dur = int(hit.get("id")), float(hit.get("duration", 0))
            except (TypeError, ValueError):
                continue
            if hid in seen or not (s["clip_min_seconds"] <= dur <= s["clip_max_seconds"]):
                continue
            best = _best_file(hit)
            if not best:
                continue
            seen.add(hid)
            found.append({
                "id": hid, "query": query, "duration": dur, "url": best["url"],
                "width": best.get("width", 0), "height": best.get("height", 0),
                "thumbnail": best.get("thumbnail", ""), "tags": hit.get("tags", ""),
                "user": hit.get("user", ""), "pageURL": hit.get("pageURL", ""),
                "likes": int(hit.get("likes", 0) or 0), "uses": uses("clips", str(hid)),
            })
        # Every query runs (results are cached a day), so the genre staples always compete.
    # Unused first, then the clips whose tags match the scene and the genre; longer loops hide the seam.
    for c in found:
        c["relevance"] = relevance(c, planned)
    found = [c for c in found if c["relevance"] > 0] or found
    found.sort(key=lambda c: (c["uses"], -c["relevance"], -c["duration"], -c["likes"]))
    return found[:count]


def fetch_clip(cfg: dict, candidate: dict) -> Path:
    from .visuals import _download
    s = settings(cfg)
    cache = ROOT / s["clip_cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"loop_{candidate['id']}_{candidate.get('width', 0)}.mp4"
    if not _download(candidate["url"], target, f"loop {candidate['id']}"):
        raise RuntimeError(f"could not download Pixabay clip {candidate['id']}")
    return target


def seamless_loop(clip: Path, out_dir: Path, cfg: dict) -> Path:
    """Re-encode the clip once so its end cross-fades into its start; repeats then join invisibly."""
    vid = cfg["video"]
    w, h = vid["resolution"]
    fps = int(vid["fps"])
    total = duration_of(clip)
    fade = min(1.0, max(0.3, total / 6))
    key = hashlib.sha256(f"{clip.resolve()}|{w}x{h}|{fps}|{fade:.2f}".encode("utf-8")).hexdigest()[:10]
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    loop = work / f"loop_{key}.mp4"
    if loop.exists():
        return loop
    frames = max(1, int(round((total - fade) * fps)))
    prep = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},setsar=1,"
            f"format=yuv420p")
    graph = (f"[0:v]{prep}[a];[1:v]{prep}[b];"
             f"[a][b]xfade=transition=fade:duration={fade:.3f}:offset={total - fade:.3f},"
             f"trim=start={fade:.3f}:end={total:.3f},setpts=PTS-STARTPTS[v]")
    log(f"lofi: making a seamless {total - fade:.0f}s loop from the Pixabay clip")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(clip), "-i", str(clip), "-filter_complex", graph,
         "-map", "[v]", "-an", "-r", str(fps), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-g", str(frames), "-keyint_min", str(frames), "-movflags", "+faststart", str(loop)])
    return loop
