"""Stage 4: visual source per scene.

Sources:
  pixabay       - free stock footage from Pixabay, cached on disk
  diffusion     - AI panels rendered locally with Stable Diffusion (see diffusion.py)
  illustration  - original flat motion-comic panels drawn locally with Pillow
  placeholder   - generated dark slides with the query text, for testing without a key
  local         - user-uploaded media files (chosen in the UI)

Pexels is intentionally not supported: new Pexels API key issuance has been
paused (since ~Sept 2026), so Pixabay is the only stock-footage source that
new users can still sign up for.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from pathlib import Path

import requests

from .config import ROOT, env
from .fonts import ffmpeg_path, find_font
from . import diffusion
from .illustrations import render_panel
from .motion import plan_shots
from .ranking import rank_scene_batches
from .util import log, read_json, run, write_json

PIXABAY_URL = "https://pixabay.com/api/videos/"
PIXABAY_IMAGE_URL = "https://pixabay.com/api/"
PIXABAY_SEARCH_CACHE_SECONDS = 24 * 60 * 60
SOURCES = ("pixabay", "diffusion", "illustration", "placeholder")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


# ---------------------------------------------------------------------------
# Candidate fetching (returns multiple options per scene for manual selection)
# ---------------------------------------------------------------------------

def fetch_candidates_for_scenes(scenes: list[dict], out_dir: Path,
                                cfg: dict, candidates_per_scene: int = 6,
                                source: str = "pixabay",
                                script_context: dict | None = None,
                                start_index: int = 0,
                                page: int = 1) -> dict[int, list[dict]]:
    """Fetch N candidate media items per scene for manual selection.

    ``start_index`` is the scene number of ``scenes[0]`` in the full script, so a
    single scene can be re-fetched without colliding with scene 0's files.
    ``page`` selects a further page of Pixabay results for "regenerate".

    Returns a dict mapping scene_index -> list of candidate dicts:
        {
            "path": Path,           # local path to the media file (video or image)
            "type": "video" | "image",
            "query": str,
            "source": "pixabay" | "illustration" | "placeholder" | "local",
            "metadata": dict        # pixabay video metadata or {}
        }
    """
    per_cut = cfg["video"]["seconds_per_cut"]
    width, height = cfg["video"]["resolution"]
    prefer_portrait = height > width
    results: dict[int, list[dict]] = {}

    for i, scene in enumerate(scenes, start_index):
        candidates = []
        query = scene.get("visual_query", "")

        if source == "illustration":
            path = render_panel(scene, out_dir, i, cfg, script_context)
            candidates.append({
                "path": path,
                "type": "image",
                "query": query,
                "source": "illustration",
                "metadata": {},
            })

        if source == "diffusion":
            if not diffusion.available():
                raise RuntimeError(diffusion.install_hint())
            per_fetch = max(1, int(diffusion.settings(cfg)["candidates"]))
            for k in range((page - 1) * per_fetch, page * per_fetch):
                candidates.append(_diffusion_candidate(scene, out_dir, i, cfg, script_context, k))

        if source == "pixabay":
            key = env("PIXABAY_API_KEY")
            if not key:
                log("pixabay: PIXABAY_API_KEY is not set; generating placeholders")
            else:
                candidates = _pixabay_candidates(scene, cfg, candidates_per_scene, key,
                                                 prefer_portrait, page)

        # Fallback: placeholder color slides if no candidates found
        if not candidates:
            if source == "pixabay" and env("PIXABAY_API_KEY"):
                log(f"pixabay: no clips for '{query}'; generating placeholders")
            # The slides differ only in shade, so a handful is plenty to pick from.
            for k in range(min(candidates_per_scene, 4)):
                path = _placeholder(scene, out_dir, i, k, per_cut, cfg)
                candidates.append({
                    "path": path,
                    "type": "video",
                    "query": query,
                    "source": "placeholder",
                    "metadata": {},
                })

        results[i] = candidates

    # Score stock candidates against their scenes, several scenes per request.
    to_rank = [i for i, cands in results.items()
               if source == "pixabay" and cands and cands[0]["source"] == "pixabay"]
    if to_rank:
        ranked = rank_scene_batches([(scenes[i - start_index], results[i]) for i in to_rank],
                                    cfg, out_dir, script_context)
        for i, cands in zip(to_rank, ranked):
            results[i] = cands

    # When the library has nothing usable for a scene, draw the scene instead.
    threshold = int(cfg["visuals"].get("diffusion_fallback_score", 0) or 0)
    if source == "pixabay" and threshold > 0 and diffusion.available():
        for i in to_rank:
            best = max((c["metadata"].get("score", -1) for c in results[i]), default=-1)
            if 0 <= best < threshold:
                log(f"visuals: scene {i + 1} stock scored {best}/10; rendering an AI panel instead")
                try:
                    cand = _diffusion_candidate(scenes[i - start_index], out_dir, i, cfg, script_context, 0)
                except Exception as exc:
                    log(f"diffusion: skipped ({type(exc).__name__}: {exc})")
                    continue
                cand["metadata"]["score"] = threshold
                cand["metadata"]["reason"] = f"AI panel; best stock match scored {best}/10"
                results[i] = [cand] + results[i]

    return results


def _diffusion_candidate(scene: dict, out_dir: Path, i: int, cfg: dict,
                         script_context: dict | None, variant: int) -> dict:
    path = diffusion.generate_panel(scene, out_dir, i, cfg, script_context, variant=variant)
    return {
        "path": path,
        "type": "image",
        "query": scene.get("visual_query", ""),
        "source": "diffusion",
        "metadata": {"duration": 0, "variant": variant},
    }


def _pixabay_candidates(scene: dict, cfg: dict, count: int, key: str,
                        prefer_portrait: bool, page: int = 1) -> list[dict]:
    """Videos and photos for every search phrasing of a scene, interleaved so the
    best-first term contributes most, de-duplicated by Pixabay page URL."""
    cache = ROOT / cfg["visuals"]["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    include_photos = bool(cfg["visuals"].get("include_photos", True))
    terms = [t for t in scene.get("search_terms", []) if t] or [scene.get("visual_query", "")]
    terms = [t for t in terms if t][:4]

    per_term = max(3, -(-count // max(1, len(terms))) + 1)
    streams: list[list[tuple[str, dict, str]]] = []
    for term in terms:
        videos = _search_pixabay_videos(term, per_term, key, cache, prefer_portrait, page=page) or []
        photos = (_search_pixabay_images(term, per_term, key, cache, prefer_portrait, page=page) or []
                  if include_photos else [])
        stream: list[tuple[str, dict, str]] = []
        # Roughly two videos for every photo, videos first.
        vi, pi = 0, 0
        while vi < len(videos) or pi < len(photos):
            for _ in range(2):
                if vi < len(videos):
                    stream.append(("video", videos[vi], term))
                    vi += 1
            if pi < len(photos):
                stream.append(("image", photos[pi], term))
                pi += 1
        streams.append(stream)

    seen: set[str] = set()
    candidates: list[dict] = []
    cursor = 0
    while len(candidates) < count and any(cursor < len(st) for st in streams):
        for stream in streams:
            if len(candidates) >= count or cursor >= len(stream):
                continue
            kind, hit, term = stream[cursor]
            page_url = hit.get("pageURL") or ""
            if page_url in seen:
                continue
            seen.add(page_url)
            cand = (_video_candidate(hit, term, cache, prefer_portrait) if kind == "video"
                    else _image_candidate(hit, term, cache))
            if cand:
                candidates.append(cand)
        cursor += 1
    return candidates


def _download(url: str, target: Path, label: str) -> bool:
    if target.exists():
        return True
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(partial, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
        partial.replace(target)
        return True
    except (requests.RequestException, OSError) as exc:
        partial.unlink(missing_ok=True)
        log(f"pixabay: download failed for '{label}': {type(exc).__name__}")
        return False


def _video_candidate(video: dict, term: str, cache: Path, prefer_portrait: bool) -> dict | None:
    best = _pick_pixabay_file(video.get("videos", {}), prefer_portrait)
    if not best:
        return None
    url = best["url"]
    target = cache / f"{hashlib.sha256(url.encode()).hexdigest()}.mp4"
    if not _download(url, target, term):
        return None
    return {
        "path": target,
        "type": "video",
        "query": term,
        "source": "pixabay",
        "metadata": {
            "user": video.get("user", ""),
            "pageURL": video.get("pageURL", ""),
            "tags": video.get("tags", ""),
            "duration": video.get("duration", 0),
            "width": best.get("width", 0),
            "height": best.get("height", 0),
        },
    }


def _image_candidate(photo: dict, term: str, cache: Path) -> dict | None:
    url = photo.get("largeImageURL") or photo.get("webformatURL")
    if not url:
        return None
    target = cache / f"{hashlib.sha256(url.encode()).hexdigest()}.jpg"
    if not _download(url, target, term):
        return None
    return {
        "path": target,
        "type": "image",
        "query": term,
        "source": "pixabay",
        "metadata": {
            "user": photo.get("user", ""),
            "pageURL": photo.get("pageURL", ""),
            "tags": photo.get("tags", ""),
            "duration": 0,
            "width": photo.get("imageWidth", 0),
            "height": photo.get("imageHeight", 0),
        },
    }


# ---------------------------------------------------------------------------
# Final plan assembly (called after user makes selections)
# ---------------------------------------------------------------------------

def clips_for_scenes(scenes: list[dict], durations: list[float], out_dir: Path,
                     cfg: dict, source: str,
                     script_context: dict | None = None,
                     selections: dict[int, str | None] | None = None,
                     custom_media: dict[int, Path | None] | None = None) -> list[list[tuple[Path, float]]]:
    """Return, per scene, a list of (clip_path, seconds_to_show).

    If ``selections`` is provided (mapping scene_index -> chosen clip path),
    those paths are used. Otherwise falls back to the configured source.
    """
    per_cut = cfg["video"]["seconds_per_cut"]
    selections = _int_keys(selections)
    custom_media = _int_keys(custom_media)
    candidate_meta = _load_candidate_metadata(out_dir)
    ranked = _load_candidates(out_dir)
    plan = []
    for i, (scene, dur) in enumerate(zip(scenes, durations)):
        cuts = max(1, math.ceil(dur / per_cut))
        seg = dur / cuts

        # Determine the clip path for this scene
        clip_path = None
        if custom_media.get(i):
            clip_path = Path(custom_media[i])
        elif selections.get(i):
            clip_path = Path(selections[i])

        if clip_path is not None and not clip_path.exists():
            log(f"visuals: selected media for scene {i + 1} is missing ({clip_path.name}); "
                "falling back to automatic selection")
            clip_path = None

        if clip_path is None and source in ("pixabay", "diffusion"):
            # No manual choice: take the best-ranked fetched candidate.
            for cand in ranked.get(i, []):
                if cand["source"] in ("pixabay", "diffusion") and Path(cand["path"]).exists():
                    clip_path = Path(cand["path"])
                    score = cand.get("metadata", {}).get("score")
                    log(f"visuals: scene {i + 1} auto-picked {clip_path.name}"
                        + (f" (score {score}/10)" if isinstance(score, int) and score >= 0 else ""))
                    break

        if clip_path is not None:
            # Credit the Pixabay creator of a hand-picked candidate.
            meta = candidate_meta.get(str(clip_path.resolve()))
            if meta and meta.get("pageURL"):
                _record_pixabay_credits(out_dir, scene.get("visual_query", ""), [meta])
            # A still becomes a run of short camera moves, cutting to the scene's other
            # good stills when it has any; a video keeps its cuts.
            if clip_path.suffix.lower() in IMAGE_SUFFIXES:
                stills = [clip_path] + _extra_stills(ranked.get(i, []), clip_path, cfg)
                plan.append(plan_shots(stills, dur, cfg, f"{script_title(script_context)}|{i}"))
            else:
                plan.append([(clip_path, seg)] * cuts)
            continue

        if source == "illustration":
            panel = render_panel(scene, out_dir, i, cfg, script_context)
            plan.append(plan_shots([panel], dur, cfg, f"{script_title(script_context)}|{i}"))
            continue
        if source == "diffusion":
            panel = diffusion.generate_panel(scene, out_dir, i, cfg, script_context)
            plan.append(plan_shots([panel], dur, cfg, f"{script_title(script_context)}|{i}"))
            continue

        paths = []
        if source == "pixabay":
            paths = _pixabay_clips(scene.get("visual_query", ""), cuts, cfg, out_dir)
        if not paths:
            paths = [_placeholder(scene, out_dir, i, 0, seg, cfg)]
        # Rotate through the fetched clips so each cut shows different footage.
        plan.append([(paths[k % len(paths)], seg) for k in range(cuts)])
    return plan


def script_title(script_context: dict | None) -> str:
    return str((script_context or {}).get("title", ""))


def _extra_stills(candidates: list[dict], chosen: Path, cfg: dict) -> list[Path]:
    """Other stills from the scene worth cutting to: AI panels, or stock that scored well."""
    limit = max(1, int(cfg.get("visuals", {}).get("stills_per_scene", 3))) - 1
    min_score = int(cfg.get("visuals", {}).get("extra_still_min_score", 6))
    chosen_key = str(Path(chosen).resolve())
    extras: list[Path] = []
    for cand in candidates:
        if len(extras) >= limit:
            break
        path = Path(cand["path"])
        if cand.get("type") != "image" or str(path.resolve()) == chosen_key or not path.exists():
            continue
        score = cand.get("metadata", {}).get("score")
        if cand.get("source") == "diffusion" or (isinstance(score, int) and score >= min_score):
            extras.append(path)
    return extras


def serialize_candidates(candidates: dict) -> dict:
    """JSON-safe form of fetch_candidates_for_scenes() output."""
    return {
        str(idx): [
            {"path": str(c["path"]), "type": c["type"], "query": c["query"],
             "source": c["source"], "metadata": c["metadata"]}
            for c in cands
        ]
        for idx, cands in candidates.items()
    }


def deserialize_candidates(raw: dict) -> dict[int, list[dict]]:
    """Inverse of serialize_candidates(); tolerates hand-edited files."""
    result: dict[int, list[dict]] = {}
    for idx, cands in (raw or {}).items():
        try:
            key = int(idx)
        except (TypeError, ValueError):
            continue
        result[key] = [
            {"path": Path(c["path"]), "type": c.get("type", "video"), "query": c.get("query", ""),
             "source": c.get("source", "local"), "metadata": c.get("metadata", {})}
            for c in (cands or []) if isinstance(c, dict) and c.get("path")
        ]
    return result


def _int_keys(mapping: dict | None) -> dict[int, object]:
    """Accept JSON (string) or Python (int) scene keys and drop empty values."""
    result: dict[int, object] = {}
    for k, v in (mapping or {}).items():
        if not v:
            continue
        try:
            result[int(k)] = v
        except (TypeError, ValueError):
            continue
    return result


def _load_candidates(out_dir: Path) -> dict[int, list[dict]]:
    path = out_dir / "media_candidates.json"
    if not path.exists():
        return {}
    try:
        return deserialize_candidates(read_json(path))
    except (OSError, ValueError):
        return {}


def _load_candidate_metadata(out_dir: Path) -> dict[str, dict]:
    """Map every cached candidate path to its Pixabay metadata, if any."""
    path = out_dir / "media_candidates.json"
    if not path.exists():
        return {}
    try:
        raw = read_json(path)
    except (OSError, ValueError):
        return {}
    meta: dict[str, dict] = {}
    for cands in (raw.values() if isinstance(raw, dict) else []):
        for c in cands or []:
            if isinstance(c, dict) and c.get("source") == "pixabay" and c.get("path"):
                meta[str(Path(c["path"]).resolve())] = {
                    "user": c.get("metadata", {}).get("user", ""),
                    "pageURL": c.get("metadata", {}).get("pageURL", ""),
                    "tags": c.get("metadata", {}).get("tags", ""),
                }
    return meta


# ---------------------------------------------------------------------------
# Pixabay internals (unchanged logic, kept for candidate fetching)
# ---------------------------------------------------------------------------

def _pixabay_clips(query: str, count: int, cfg: dict, out_dir: Path) -> list[Path]:
    key = env("PIXABAY_API_KEY")
    if not key:
        return []
    cache = ROOT / cfg["visuals"]["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    width, height = cfg["video"]["resolution"]
    prefer_portrait = height > width
    videos = _search_pixabay_videos(query, count, key, cache, prefer_portrait)
    if videos is None:
        return []

    paths = []
    used_videos = []
    for video in videos:
        if len(paths) >= count:
            break
        best = _pick_pixabay_file(video.get("videos", {}), prefer_portrait)
        if not best:
            continue
        url = best["url"]
        target = cache / f"{hashlib.sha256(url.encode()).hexdigest()}.mp4"
        if not target.exists():
            log(f"pixabay: downloading clip for '{query}'")
            partial = target.with_suffix(".part")
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(partial, "wb") as f:
                        for chunk in r.iter_content(1 << 16):
                            f.write(chunk)
                partial.replace(target)
            except requests.RequestException as exc:
                partial.unlink(missing_ok=True)
                log(f"pixabay: download failed: {type(exc).__name__}")
                continue
        paths.append(target)
        used_videos.append(video)
    if used_videos:
        _record_pixabay_credits(out_dir, query, used_videos)
    return paths


def _search_pixabay_videos(query: str, count: int, key: str, cache: Path,
                           prefer_portrait: bool = False, page: int = 1) -> list[dict] | None:
    cache.mkdir(parents=True, exist_ok=True)
    per_page = min(max(count, 10), 20)
    page = max(1, int(page))
    orientation = "portrait" if prefer_portrait else "landscape"
    cache_id = hashlib.sha256(
        f"{key}\0{query}\0{per_page}\0{orientation}\0{page}".encode()).hexdigest()
    cache_path = cache / f"search_{cache_id}.json"

    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < PIXABAY_SEARCH_CACHE_SECONDS:
        hits = _cached_pixabay_hits(cache_path)
        if isinstance(hits, list):
            return hits

    try:
        resp = requests.get(PIXABAY_URL, params={
            "key": key,
            "q": query,
            "safesearch": "true",
            "min_width": 360 if prefer_portrait else 640,
            "min_height": 640 if prefer_portrait else 360,
            "per_page": per_page,
            "page": page,
        }, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        hits = result.get("hits", []) if isinstance(result, dict) else []
        if not isinstance(hits, list):
            hits = []
        write_json(cache_path, {"hits": hits})
        return hits
    except (requests.RequestException, ValueError) as exc:
        detail = _pixabay_error_detail(exc, key)
        if cache_path.exists():
            hits = _cached_pixabay_hits(cache_path)
            if isinstance(hits, list):
                log(f"pixabay: search failed for '{query}' ({detail}); using cached results")
                return hits
        log(f"pixabay: search failed for '{query}' ({detail})")
        return None


def _search_pixabay_images(query: str, count: int, key: str, cache: Path,
                           prefer_portrait: bool = False, page: int = 1) -> list[dict] | None:
    cache.mkdir(parents=True, exist_ok=True)
    per_page = min(max(count, 3), 20)
    page = max(1, int(page))
    orientation = "vertical" if prefer_portrait else "horizontal"
    cache_id = hashlib.sha256(
        f"img\0{key}\0{query}\0{per_page}\0{orientation}\0{page}".encode()).hexdigest()
    cache_path = cache / f"search_{cache_id}.json"

    if cache_path.exists() and time.time() - cache_path.stat().st_mtime < PIXABAY_SEARCH_CACHE_SECONDS:
        hits = _cached_pixabay_hits(cache_path)
        if isinstance(hits, list):
            return hits

    try:
        resp = requests.get(PIXABAY_IMAGE_URL, params={
            "key": key,
            "q": query,
            "image_type": "photo",
            "orientation": orientation,
            "safesearch": "true",
            "min_width": 720 if prefer_portrait else 1280,
            "min_height": 1280 if prefer_portrait else 720,
            "per_page": per_page,
            "page": page,
        }, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        hits = result.get("hits", []) if isinstance(result, dict) else []
        if not isinstance(hits, list):
            hits = []
        write_json(cache_path, {"hits": hits})
        return hits
    except (requests.RequestException, ValueError) as exc:
        detail = _pixabay_error_detail(exc, key)
        if cache_path.exists():
            hits = _cached_pixabay_hits(cache_path)
            if isinstance(hits, list):
                return hits
        log(f"pixabay: photo search failed for '{query}' ({detail})")
        return None


def _pixabay_error_detail(exc: Exception, key: str) -> str:
    response = getattr(exc, "response", None)
    if response is None:
        return type(exc).__name__
    status = f"HTTP {response.status_code}"
    body = (response.text or "").strip()
    body = body.replace(key, "[redacted]")
    body = re.sub(r"(?i)([?&]key=)[^&\s\"']+", r"\1[redacted]", body)
    body = " ".join(body.split())[:180]
    return f"{status}: {body}" if body else status


def _cached_pixabay_hits(path: Path) -> list[dict] | None:
    try:
        cached = read_json(path)
    except (OSError, ValueError):
        return None
    hits = cached.get("hits") if isinstance(cached, dict) else None
    return hits if isinstance(hits, list) else None


def _pick_pixabay_file(files: dict, prefer_portrait: bool = False) -> dict | None:
    if not isinstance(files, dict):
        return None
    candidates = [
        f for f in files.values()
        if isinstance(f, dict) and f.get("url") and f.get("width") and f.get("height")
        and (f["height"] > f["width"]) == prefer_portrait
    ]
    if not candidates:
        candidates = [
            f for f in files.values()
            if isinstance(f, dict) and f.get("url") and f.get("width") and f.get("height")
        ]
    for name in ("large", "medium", "small", "tiny"):
        chosen = files.get(name)
        if chosen in candidates:
            return chosen
    return max(candidates, key=lambda f: f["width"], default=None)


def _record_pixabay_credits(out_dir: Path, query: str, videos: list[dict]) -> None:
    path = out_dir / "pixabay_credits.json"
    credits = read_json(path) if path.exists() else []
    known = {entry.get("page_url") for entry in credits}
    for video in videos:
        page_url = video.get("pageURL")
        if not page_url or page_url in known:
            continue
        credits.append({
            "creator": video.get("user", "Pixabay contributor"),
            "page_url": page_url,
            "tags": video.get("tags", ""),
            "search_query": query,
        })
        known.add(page_url)
    write_json(path, credits)


def _placeholder(scene: dict, out_dir: Path, i: int, k: int, seconds: float, cfg: dict) -> Path:
    clips = out_dir / "placeholders"
    clips.mkdir(exist_ok=True)
    path = clips / f"scene_{i:02d}_{k}.mp4"
    if path.exists():
        return path
    w, h = cfg["video"]["resolution"]
    label = _drawtext_label(scene.get("visual_query", "") or f"scene {i + 1}")
    shade = ["0x1b2838", "0x2a1b38", "0x1b3828", "0x38261b"][(i + k) % 4]
    base = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
            "-i", f"color=c={shade}:s={w}x{h}:d={seconds:.2f}:r={cfg['video']['fps']}"]
    tail = ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(path)]
    font = find_font(bold=False)
    font_arg = f":fontfile='{ffmpeg_path(font)}'" if font else ""
    text_filter = (f"drawtext=text='{label}'{font_arg}:fontsize=56:fontcolor=white@0.5"
                   f":x=(w-text_w)/2:y=(h-text_h)/2")
    try:
        run(base + ["-vf", text_filter] + tail)
    except RuntimeError:
        log("placeholder: drawtext unavailable in this ffmpeg build, using plain slides")
        run(base + tail)
    return path


def _drawtext_label(text: str) -> str:
    """Strip characters that break ffmpeg's filter parser or drawtext escaping."""
    cleaned = re.sub(r"[\\'\":;,\[\]%=]", " ", text)
    return " ".join(cleaned.split())[:60]


def extract_thumbnail(media_path: Path, out_path: Path, at_seconds: float = 0.5) -> bool:
    """Extract a single frame from a video or image file as a thumbnail JPG."""
    seek = [] if Path(media_path).suffix.lower() in IMAGE_SUFFIXES else ["-ss", str(at_seconds)]
    try:
        run(["ffmpeg", "-y", "-loglevel", "error", *seek,
             "-i", str(media_path), "-frames:v", "1", "-q:v", "3", str(out_path)])
        return out_path.exists()
    except RuntimeError:
        return False
