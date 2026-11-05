"""Smarter stock selection: several search phrasings per scene, and a vision
model that scores each candidate thumbnail against the scene description.

Both steps use the free tier of the configured LLM (Gemini Flash by default
for images) and degrade gracefully: without a key or on any API error the
pipeline keeps the plain keyword search and the original candidate order.

Free tiers are counted in requests per day, so ranking packs several scenes
into one request (``visuals.rank_batch_scenes``, default 3): a six-scene
short costs two ranking calls plus one search-term call.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .llm import generate_json, vision_available
from .util import log, run, write_json

TERMS_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "terms": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["terms"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenes"],
    "additionalProperties": False,
}

RANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame": {"type": "integer"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["frame", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

RANK_THUMB_WIDTH = 512
RANK_TIMEOUT_SECONDS = 75
LOW_MATCH_SCORE = 4  # below this, stock footage is a poor fit and the UI says so


def _clean_term(term: str) -> str:
    term = re.sub(r"[^A-Za-z0-9 '\-]+", " ", str(term))
    return " ".join(term.split())[:100]


def ensure_search_terms(script: dict, cfg: dict, out_dir: Path | None = None) -> dict:
    """Give every scene a ``search_terms`` list and persist it to script.json.

    One LLM call per video. Stock libraries index literal, visible things, so the
    prompt asks for concrete nouns a photographer would have tagged, not themes.
    """
    scenes = script.get("scenes", [])
    per_scene = int(cfg.get("visuals", {}).get("search_terms_per_scene", 3) or 0)
    if per_scene <= 1 or all(scene.get("search_terms") for scene in scenes):
        for scene in scenes:
            scene.setdefault("search_terms", [scene.get("visual_query", "")] if scene.get("visual_query") else [])
        return script

    system = (
        "You write search queries for free stock footage and photo libraries (Pixabay). "
        "Stock libraries only match literal, visible subjects that a photographer tagged: "
        "objects, animals, places, weather, people doing everyday actions. They never match "
        "story names, emotions, or abstract ideas. Every query must be in English, 1 to 3 words, "
        "and depict the scene even if only approximately. Keep the era and culture of the scene "
        "in the query whenever it matters: write 'ancient warrior' or 'medieval castle', never a "
        "bare 'soldier' or 'meeting' that would return modern offices, firearms, or jets. "
        "Prefer timeless subjects (landscapes, fire, stone, animals, hands, crafts, nature) when "
        "the exact subject is unlikely to exist as stock footage."
    )
    lines = []
    for i, scene in enumerate(scenes, 1):
        lines.append(
            f"Scene {i}: narration: {scene.get('narration', '')}\n"
            f"  visual description: {scene.get('visual_description', '')}\n"
            f"  setting: {scene.get('setting', '')}; subject: {scene.get('visual_subject', '')}; "
            f"action: {scene.get('action', '')}; props: {', '.join(scene.get('visual_objects', []))}"
        )
    prompt = (
        f"Video title: {script.get('title', '')}\n"
        f"Format: {script.get('content_type', '')}\n\n"
        + "\n".join(lines)
        + f"\n\nFor each scene, in order, give {per_scene} different stock search queries, best first. "
        "Mix one literal query (the main subject with its era), one about the setting or environment, "
        "and one close-up of a prop or natural detail. Avoid names of people or fictional characters. "
        f"Return exactly {len(scenes)} scene entries."
    )
    try:
        result = generate_json(system, prompt, TERMS_SCHEMA, max_tokens=4000, effort="medium")
        entries = result.get("scenes", [])
    except Exception as exc:  # any provider error: keep the single query
        log(f"search terms: using the script's visual_query only ({type(exc).__name__}: {exc})")
        entries = []

    for i, scene in enumerate(scenes):
        base = scene.get("visual_query", "")
        terms = []
        if i < len(entries):
            terms = [_clean_term(t) for t in entries[i].get("terms", []) if _clean_term(t)]
        if base and base not in terms:
            terms.append(base)
        scene["search_terms"] = terms[:per_scene + 1] or [base]
    if out_dir is not None:
        write_json(out_dir / "script.json", script)
    log("search terms: " + " | ".join(", ".join(s.get("search_terms", [])) for s in scenes))
    return script


def _rank_thumbnail(media_path: Path, out_dir: Path) -> bytes | None:
    """A small JPEG of a candidate (first frame for video), cached per file."""
    thumbs = out_dir / "work" / "rank_thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(media_path.resolve()).encode("utf-8")).hexdigest()[:16]
    thumb = thumbs / f"{key}.jpg"
    if not thumb.exists():
        is_image = media_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        seek = [] if is_image else ["-ss", "0.5"]
        try:
            run(["ffmpeg", "-y", "-loglevel", "error", *seek, "-i", str(media_path),
                 "-frames:v", "1", "-vf", f"scale={RANK_THUMB_WIDTH}:-2", "-q:v", "6", str(thumb)])
        except RuntimeError:
            return None
    try:
        return thumb.read_bytes()
    except OSError:
        return None


def _scene_brief(scene: dict) -> str:
    return (
        f"narration: {scene.get('narration', '')}\n"
        f"    wanted visual: {scene.get('visual_description', '')}\n"
        f"    setting: {scene.get('setting', '')}. subject: {scene.get('visual_subject', '')}. "
        f"action: {scene.get('action', '')}. props: {', '.join(scene.get('visual_objects', []))}"
    )


def rank_scene_batches(items: list[tuple[dict, list[dict]]], cfg: dict, out_dir: Path,
                       script_context: dict | None = None) -> list[list[dict]]:
    """Score every scene's candidates 0-10 with a vision model; best first.

    ``items`` pairs each scene with its candidates. Scenes are packed several per
    request to stay inside daily free-tier request limits. Scores and one-line
    reasons land in each candidate's metadata; unscorable candidates keep -1 and
    sort last. On any failure the affected scenes keep their search order.
    """
    results = [cands for _, cands in items]
    visuals_cfg = cfg.get("visuals", {})
    if not items or not visuals_cfg.get("rank_with_llm", True):
        return results
    if not vision_available():
        log("rank: no vision-capable API key configured; keeping search order")
        return results

    batch_size = max(1, int(visuals_cfg.get("rank_batch_scenes", 3) or 1))
    # Free tiers count requests per day; a long video has dozens of scenes, so a
    # format can cap how many ranking calls one video may spend (0 = no cap).
    max_requests = int(visuals_cfg.get("rank_max_requests", 0) or 0)
    requests_made = 0
    context = script_context or {}
    w, h = cfg["video"]["resolution"]
    frame_kind = "widescreen" if w >= h else "vertical short"
    for start in range(0, len(items), batch_size):
        batch = list(range(start, min(start + batch_size, len(items))))
        if max_requests and requests_made >= max_requests:
            log(f"rank: request budget of {max_requests} used; scenes {batch[0] + 1}-{len(items)} keep search order")
            break
        images: list[bytes] = []
        frame_owner: list[tuple[int, dict]] = []  # frame number -> (item index, candidate)
        scene_lines = []
        for item_index in batch:
            scene, cands = items[item_index]
            first = len(images) + 1
            for c in cands:
                if c.get("source") not in ("pixabay", "local"):
                    continue
                data = _rank_thumbnail(Path(c["path"]), out_dir)
                if data:
                    images.append(data)
                    frame_owner.append((item_index, c))
            last = len(images)
            if last >= first:
                scene_lines.append(f"Scene {item_index + 1} = frames {first} to {last}:\n    {_scene_brief(scene)}")
        if len(images) < 2:
            continue

        system = (
            f"You are a picture editor choosing footage for a {frame_kind} video. You will see "
            f"{len(images)} candidate frames, numbered 1 to {len(images)} in the order given. Each frame "
            "belongs to exactly one scene (the ranges are listed). Score how well each frame depicts "
            "its own scene for a viewer: subject, setting, mood, and era all matter. Penalize unrelated "
            "subjects, visible text or logos, watermarks, modern objects in historical scenes, and "
            "frames that would look odd next to the narration. Score 0-10 where 10 is a perfect "
            "depiction, 5 is a plausible generic backdrop, 0 is unusable."
        )
        prompt = (
            f"Video: {context.get('title', '')} ({context.get('content_type', '')})\n\n"
            + "\n".join(scene_lines)
            + f"\n\nReturn one entry per frame (frame 1 to {len(images)}) with an integer score and a "
            "reason of at most 12 words."
        )
        try:
            requests_made += 1
            result = generate_json(system, prompt, RANK_SCHEMA, max_tokens=4000, effort="medium",
                                   images=images, timeout=RANK_TIMEOUT_SECONDS)
            scores = result.get("scores", [])
        except Exception as exc:
            log(f"rank: scenes {batch[0] + 1}-{batch[-1] + 1} keep search order ({type(exc).__name__}: {exc})")
            continue

        for item_index in batch:
            for c in items[item_index][1]:
                c.setdefault("metadata", {})
                c["metadata"]["score"] = -1
                c["metadata"]["reason"] = ""
        for entry in scores:
            try:
                frame = int(entry.get("frame", 0)) - 1
                score = max(0, min(10, int(entry.get("score", 0))))
            except (TypeError, ValueError):
                continue
            if 0 <= frame < len(frame_owner):
                _, cand = frame_owner[frame]
                cand["metadata"]["score"] = score
                cand["metadata"]["reason"] = str(entry.get("reason", ""))[:120]
        for item_index in batch:
            scene, cands = items[item_index]
            ranked = sorted(cands, key=lambda c: c["metadata"].get("score", -1), reverse=True)
            results[item_index] = ranked
            best = ranked[0]["metadata"] if ranked else {}
            note = "" if best.get("score", -1) >= LOW_MATCH_SCORE else " (poor stock match; consider an upload or illustration)"
            log(f"rank: scene {item_index + 1} best {best.get('score')}/10 for "
                f"'{scene.get('visual_query', '')}': {best.get('reason', '')}{note}")
    return results


def rank_candidates(scene: dict, candidates: list[dict], cfg: dict, out_dir: Path,
                    script_context: dict | None = None) -> list[dict]:
    """Rank one scene's candidates (see rank_scene_batches)."""
    return rank_scene_batches([(scene, candidates)], cfg, out_dir, script_context)[0]
